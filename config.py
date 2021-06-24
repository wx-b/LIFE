import configargparse

def get_model_args():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--model', type=str, default="./model/pretrain.pth", help="choose the trained model")

    args = parser.parse_known_args()[0]
    return args

def get_demo_args():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--model', type=str, default="./model/pretrain.pth", help="choose the trained model")
    parser.add_argument('--movie_start_idx', type=int, default=2017)
    parser.add_argument('--video_start_idx', type=int, default=0)
    parser.add_argument('--frame_number', type=int, default=304)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--data_path', type=str, default="./demo/data/")

    args = parser.parse_known_args()[0]
    return args