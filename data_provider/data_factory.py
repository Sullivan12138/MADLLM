from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, WADILoader
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'WADI': WADILoader
}


def data_provider(args, flag):
    if "SMD" in args.data:
        Data = data_dict['SMD']
        file = args.data.strip('SMD')
    else:
        Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    
    drop_last = False
    if "SMD" in args.data:
        data_set = Data(
            root_path=args.root_path,
            file = file,
            win_size=args.seq_len,
            few_shot=args.few_shot,
            flag=flag,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            few_shot=args.few_shot,
            flag=flag,
        )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
