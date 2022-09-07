import itertools
import os
from collections import OrderedDict
from scipy.stats import loguniform
import pdb
from math import exp
import subprocess

def gen_args(params):
	args = [
		f'train',
		f'--batch_size={params["batch_size"]}',
		f'--box_type=BoxTensor',
		f'--data_device=gpu',
		#f'--dataset=wackypedia_lemma',
		f'--dataset=ptb',
		f'--embedding_dim=64',
		f'--eval_file=./data/similarity_datasets/',
		f'--int_temp=1.9678289474987882',
		f'--log_frequency=10',
		f'--loss_fn=max_margin',
		f'--lr={params["lr"]}',
		f'--margin=5',
		f'--model_type=Word2BoxConjunction',
		f'--n_gram={params["window_size"]}',
		f'--negative_samples={params["negative_samples"]}',
		f'--num_epochs=10',
		f'--subsample_thresh=0.001',
		f'--vol_temp=0.33243242379830407',
		f'--save_model',
		f'--add_pad',
		f'--save_dir=./experiments/default', #change this
		f'--subsample_thresh={params["subsampling_threshold"]}'
	]

	return args

def main():

	#create hyper-parameter lists
	grid = OrderedDict(
		batch_size=[2**11, 2**12, 2**13, 2**14, 2**15],
		lr=loguniform.pdf([exp(-1), exp(-10)], exp(-10), exp(-1)), #pdf means probability density function
		window_size=[5,6,7,8,9,10],
		negative_samples=[2,5,10,2],
		subsampling_threshold=[1e-3, 1e-4]
	)
	keys=list(grid.keys())

	#begin grid search
	for params in itertools.product(*tuple(grid.values())):

		#create parameter dictionary
		params_dict={keys[i]:params[i] for i in range(len(params))}

		#generate arguements string
		args=gen_args(params_dict)

		#start training process for given parameters
		process=subprocess.Popen(['language-modeling-with-boxes'] + args)
		process.wait()
		#break
				
	print('done')


if __name__ == '__main__':
	main()
