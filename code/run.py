import os
import sys
import argparse
class Loader(object):
    """
    Desc:
        Dynamically Load a Module
    """
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def load(path):
        """
        Args:
            path to the script
        """
        try:
            items = os.path.split(path)
            sys.path.append(os.path.join(*items[:-1]))
            ip_module = __import__(items[-1][:-3])
            return ip_module
        except Exception as error:
            print("IMPORT ERROR: ", error)
            print("Load module [path %s] error: %s" % (path, traceback.format_exc()))
            traceback.print_exc()
            return None

def run(args):
    print(args)

    if (args.need_fh and args.retrain_fh):
        fh = Loader.load('attn1.py')
        print('running fh')
        fh.main()
    fname1 = args.fs + '_select_single.py'
    print('running fs:', fname1)
    fs = Loader.load(fname1)
    fs.main()
    fname2 = args.fp + '_predict_'
    if (args.need_fh):
        fname2 = fname2 + 'comb'
    else:
        fname2 = fname2 + 'single'
    if (args.fs == 'cnn' or args.fs == 'dnn' or args.fs == 'lstm'):
        fname2 = fname2 + '_deep'
    fname2 = fname2 + '.py'
    fp = Loader.load(fname2)
    print('running fp:', fname2)
    fp.main(args.fs, args.saving_path)

def get_opt():
    parser = argparse.ArgumentParser(description='BBL-WPP')
    parser.add_argument('--fs', type=str, default='lasso', help='module name of fs')
    parser.add_argument('--fp', type=str, default='lasso', help='module name of fp')
    parser.add_argument('--saving_path', type=str, default='prediction', help='the path to save the results')
    parser.add_argument('--need_fh', action='store_true', help='if you need high resolution data')
    parser.add_argument('--retrain_fh', action='store_true', help='if you need to retrain fh')
    args = parser.parse_args()
    print(args)
    print(args.need_fh)
    return args

if __name__ == '__main__':
    run(get_opt())

