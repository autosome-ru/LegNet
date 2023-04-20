import argparse
from prediction_config import PredictionConfig

parser = argparse.ArgumentParser(description="script to predict expression for a given sequences file")

parser.add_argument("--seqs", type=str, required=True)
parser.add_argument("--pred_col", type=str, required=True)
parser.add_argument("--experiment_path", type=str, required=True)

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--singleton_mode",
                    choices=["infer", 'all_1', 'all_0'],
                    type=str, 
                    required=True)
parser.add_argument("--device", type=int, required=True)
   
args = parser.parse_args()


pred_conf = PredictionConfig(seqs_path=args.seqs,
                             pred_col=args.pred_col,
                             experiment_path=args.experiment_path,
                             model_path=args.model_path,
                             singleton_mode=args.singleton_mode,
                             devices=[args.device],
                             out_dir=args.out_dir)

pred_conf.run()

