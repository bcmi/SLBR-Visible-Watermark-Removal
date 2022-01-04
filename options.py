
import src.networks as networks

model_names = sorted(name for name in networks.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networks.__dict__[name]))
    
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # Model structure
        parser.add_argument('--nets', '-n', metavar='NET', default='dhn',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')
                                
        parser.add_argument('--models', '-m', metavar='NACHINE', default='basic')
        # Training strategy
        parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=30, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                            help='test batchsize')
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,metavar='LR', help='initial learning rate')
        parser.add_argument('--dlr', '--dlearning-rate', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--beta1', default=0.9, type=float, help='initial learning rate')
        parser.add_argument('--beta2', default=0.999, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')
        # Data processing
        parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                            help='flip the input during validation')

        parser.add_argument('--lambda_l1', type=float, default=4, help='the weight of L1.')
        parser.add_argument('--lambda_primary', type=float, default=0.01, help='the weight of primary mask prediction.')
        parser.add_argument('--lambda_style', default=0, type=float,
                            help='preception loss')
        parser.add_argument('--lambda_content', default=0, type=float,
                            help='preception loss')

        parser.add_argument('--lambda_iou', default=0, type=float,help='msiou loss')
        parser.add_argument('--lambda_mask', default=1, type=float,help='mask loss')
        
        parser.add_argument('--sltype', default='vggx', type=str)

        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Groundtruth Gaussian sigma.')
        parser.add_argument('--sigma-decay', type=float, default=0,
                            help='Sigma decay rate for each epoch.')
        # Miscs
        parser.add_argument('--dataset_dir', default='/PATH_TO_DATA_FOLDER/', type=str, metavar='PATH')
        parser.add_argument('--test_dir', default='/PATH_TO_DATA_FOLDER/', type=str, metavar='PATH')

        parser.add_argument('--data', default='', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        
        parser.add_argument('-da', '--data-augumentation', default=False, type=bool,
                            help='preception loss')
        parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                            help='show intermediate results')
        parser.add_argument('--input-size', default=256, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--freq', default=-1, type=int, metavar='N',
                            help='evaluation frequence')
        parser.add_argument('--normalized-input', default=False, type=bool,
                            help='train batchsize')
        parser.add_argument('--res', default=False, type=bool,help='residual learning for s2am')
        parser.add_argument('--requires-grad', default=False, type=bool,
                            help='train batchsize')
       
        parser.add_argument('--gpu',default=True,type=bool)
        parser.add_argument('--gpu_id',default='0',type=str)
        parser.add_argument('--preprocess',default='resize_crop',type=str)
        parser.add_argument('--crop_size',default=256,type=int)
        parser.add_argument('--no_flip',action='store_true')
        parser.add_argument('--masked',default=False,type=bool)
        parser.add_argument('--gan-norm', default=False,type=bool, help='train batchsize')
        parser.add_argument('--hl', default=False,type=bool, help='homogenious leanring')
        parser.add_argument('--loss-type', default='l2',type=str, help='train batchsize')

        parser.add_argument('--dataset', default='clwd',type=str, help='train batchsize')
        parser.add_argument('--name', default='v2',type=str, help='train batchsize')

        parser.add_argument('--sim_metric', default='cos',type=str, help='train batchsize')
        parser.add_argument('--k_center', default=1,type=int, help='train batchsize')
        parser.add_argument('--project_mode', default='simple',type=str, help='train batchsize')
        parser.add_argument('--mask_mode', default='cat',type=str, help='train batchsize') # vanilla, cat, ca, psp
        parser.add_argument('--bg_mode', default='res_mask',type=str, help='train batchsize') # vanilla, res_mask, res_feat, proposed
        parser.add_argument('--use_refine', action='store_true', help='train batchsize') 
        parser.add_argument('--k_refine', default=3, type=int, help='train batchsize')
        parser.add_argument('--k_skip_stage', default=3, type=int, help='train batchsize')

        return parser