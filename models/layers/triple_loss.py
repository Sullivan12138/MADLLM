import numpy
import torch 

class TripleLoss(torch.nn.modules.loss._Loss):
    def __init__(self, nb_random_samples, negative_penalty):
        super(TripleLoss, self).__init__()
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty


    def forward(self, batch, encoder):
        # batch: (batch size x nvars x num_patch x patch_length)
        batch_size, nvars, patch_num, patch_length = batch.shape

        samples = numpy.random.choice(
            nvars, size=(self.nb_random_samples, nvars)
        )

        samples = torch.LongTensor(samples)

        beginning_samples_anc = numpy.random.randint(
            0, high = patch_num, size = nvars
        )

        beginning_samples_pos = numpy.zeros(nvars, dtype=int)

        for i in range(nvars):
            beginning_samples_pos[i] =  numpy.random.randint(0, high = patch_num)
            while beginning_samples_pos[i] == beginning_samples_anc[i]:
                beginning_samples_pos[i] =  numpy.random.randint(0, high = patch_num)

        beginning_samples_neg = numpy.random.randint(
            0, high = patch_num, size = (self.nb_random_samples, nvars)
        )

        # representation: (batch size x d_model x patch_length)
        input = torch.cat(
            [batch[:,
                j: j + 1,
                beginning_samples_anc[j]: beginning_samples_anc[j] + 1, :
            ] for j in range(nvars)], dim=1
        )
        input = input.squeeze(2)
        representation = encoder(input)  # Anchors representations
        # print("representation:", representation.shape)
        
        # positive_representation: (batch size x d_model x patch_length)
        # print(type(beginning_samples_pos[0]))
        input = torch.cat(
            [batch[:,
                j: j + 1,
                beginning_samples_pos[j]: beginning_samples_pos[j] + 1, :
            ] for j in range(nvars)], dim=1
        )
        input = input.squeeze(2)
        positive_representation = encoder(input)

        d_model = representation.shape[1]
        representation = representation.permute(0, 2, 1)
        positive_representation = positive_representation.permute(0, 2, 1)
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.reshape(batch_size * patch_length, 1, d_model),
            positive_representation.reshape(batch_size * patch_length, d_model, 1)
        )))

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
                input = torch.cat(
                    [batch[:,
                        samples[i][j] : samples[i][j]+1,
                        beginning_samples_neg[i][j] : beginning_samples_neg[i][j] + 1, :
                    ] for j in range(nvars)], dim=1
                )
                input = input.squeeze(2)
                negative_representation = encoder(input)
                negative_representation = negative_representation.permute(0, 2, 1)

                loss += multiplicative_ratio * -torch.mean(
                    torch.nn.functional.logsigmoid(-torch.bmm(
                        representation.reshape(batch_size * patch_length, 1, d_model),
                        negative_representation.reshape(
                            batch_size * patch_length, d_model, 1
                        )
                    ))
                )

        return loss