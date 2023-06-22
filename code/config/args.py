

def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')


    argparser.add_argument('-m', '--method', default='sac', type=str, help='[Method] Method to use.')
    argparser.add_argument('--evaluate', action='store_true', help='evaluate models (default: False)')
    argparser.add_argument('--num-episodes', default=200, type=int, help='number of episodes.')
    argparser.add_argument('--num-vehicles', default=None, type=int, help='number of vehicles (only for evaluate).')

    argparser.add_argument('--model-dir', default='None', type=str, help='[Model] dir contains model (default: False)')
    argparser.add_argument('--model-num', default=-1, type=str, help='[Model] model-num to use.')
    argparser.add_argument('--model-index', default=None, type=str, help='[Model] model-index to use.')

    ### env param
    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--step-reset', default=0, type=int, help='')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')
    argparser.add_argument('--invert', action='store_true', help='invert axis (default: False)')
    argparser.add_argument('--render-save', action='store_true', help='save render (default: False)')

    argparser.add_argument('version', help='')

    args = argparser.parse_args()
    return args

