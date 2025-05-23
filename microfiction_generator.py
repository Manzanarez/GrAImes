# Program based on https://colab.research.google.com/drive/1_B6LKFO0khzTF319gpqIGecT5laxR_YL
# used on Rosa Espino's workshop given by Ivan Vladimir Meza

# Directories in the following variables have to be changed to where this program is going to run
# command = f"python /home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py
# model_name_or_path='/home/gerardo/Documents/GAM/GPT-2/gpt2-small-spanish
# train_file='/home/gerardo/Documents/GAM/LIPN_GitLab/data/microfictions/es/parsehub/CiudadSeva/T_cs_mf_txt_all_train.txt'
# output_dir='/home/gerardo/Documents/GAM/LIPN_GitLab/experiments/monterroso.v0.18oct2021/es/output/pass_1'
# validation_file= '/content/gdrive/MyDrive/microfiction/V_cs_mf_txt_all_validate.txt'


from __future__ import print_function
import subprocess
import transformers
import sys, stat
import time, os
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from transformers import pipeline
import re
import torch
import tqdm
from tqdm import *

#File was downloaded with wget using terminal
#wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/language-modeling/run_clm.py


#ver2dir=dict([(None,""),("master","pytorch/"),("v4.9.0","pytorch/")])
#ver2dir=dict([(None,""),("master","pytorch/"),("4.12.0","pytorch/")])

#ver2dir=dict([(None,""),("master","pytorch/"),("v4.9.0","pytorch/"),("4.12.0","pytorch/")])
#@interact(ver=ver2dir.keys())
#def download_files(ver=None):
#  if ver is None:
#    return "Select a version of the script to download"
#  url_code=f"https://raw.githubusercontent.com/huggingface/transformers/{ver}/examples/{ver2dir[ver]}language-modeling/run_clm.py"
#  wget -O run_clm.py {url_code}

#  url_code=f"https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/language-modeling/run_clm.py"

#dir="/content/gdrive/MyDrive/microfiction"
#dir="/home/gerardo/Documents/GAM/LIPN_GitLab/data/microfictions/es/parsehub/CiudadSeva/"
dir="/users/gerardo.aleman/microfiction/data/microfictions/es/parsehub/CiudadSeva"


validation_files=[os.path.join(dir,f) for f in os.listdir(dir) if f.startswith("V")]
training_files=[os.path.join(dir,f) for f in os.listdir(dir) if f.startswith("T")]
#!mkdir output

def training_model(model_name_or_path=None, \
                   do_train=True, \
                   train_file=None, \
                   block_size=1, \
                   per_device_eval_batch_size=1,
                   per_device_train_batch_size=1,
                   num_train_epochs=2, \
                   overwrite_output_dir=True, \
                   adam_beta1=0.99, \
                   adam_beta2=0.999, \
                   adam_epsilon=1e-4, \
                   max_grad_norm=1.0, \
                   metric_for_best_model=True, \
                   prediction_loss_only=True, \
                   save_total_limit=0, \
                   validation_file= None, \
                   do_eval=False, \
                   ignore_data_skip=True, \
                   output_dir=None, \
                   overwrite_cache=False \
                   ):
    start_time=time.time()
    elapsed_time= lambda: time.time() - start_time
    if do_train and model_name_or_path is None:
        print(f"Elapsed time:",elapsed_time())
        return "Select a model"
    elif do_train and train_file is None:
        print(f"Elapsed time:",elapsed_time())
        return "Select a training file"
    elif do_eval and validation_file is None:
        print(f"Elapsed time:",elapsed_time())
        return "Select a validation file"
    args=f"""--model_name_or_path {model_name_or_path} \
        --disable_tqdm True\
        --block_size {block_size}\
        --num_train_epochs {num_train_epochs}\
        --overwrite_output_dir {overwrite_output_dir}\
        --adam_beta1  {adam_beta1}\
        --adam_beta2  {adam_beta2}\
        --adam_epsilon {adam_epsilon} \
        --max_grad_norm  {max_grad_norm}\
        --metric_for_best_model {metric_for_best_model}\
        --prediction_loss_only {prediction_loss_only}\
        --save_total_limit  {save_total_limit}\
        --ignore_data_skip {ignore_data_skip} \
        --output_dir  {output_dir} """
    if overwrite_cache:
        args=args+""" --overwrite_cache\
        """
    if do_train:
        args=args+f""" --train_file  {train_file} \
        --do_train True"""
    if do_eval:
        args=args+f""" --validation_file  {validation_file} \
        --do_eval True"""
 ## !python run_clm.py {args}
#    pid = subprocess.call("/home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py args")
#    os.chmod("/home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py", stat.S_IXOTH)
#    os.chmod("/home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py", stat.S_IEXEC)
    print(f"Elapsed time 0:",elapsed_time())
    print(args)
#    return_value = os.system('/home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py' , 'args')
#    return_value = subprocess.Popen(["python /home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py {args}"], shell=True)
#    return_value = subprocess.run([sys.executable,"/home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py","args"])
#    return_value = subprocess.Popen(["python /home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py args"], shell=True)

# File that runs transformers, it is an standard file
#    command = f"python /home/gerardo/Documents/GAM/LIPN_GitLab/code/microfiction_generator/run_clm.py  \
    command = f"python3  /users/gerardo.aleman/microfiction/code/microfiction_generator/run_clm.py\
        --model_name_or_path {model_name_or_path} \
        --disable_tqdm True\
        --block_size {block_size} \
        --num_train_epochs {num_train_epochs}\
        --overwrite_output_dir {overwrite_output_dir}\
        --adam_beta1  {adam_beta1}\
        --adam_beta2  {adam_beta2}\
        --adam_epsilon {adam_epsilon} \
        --max_grad_norm  {max_grad_norm}\
        --metric_for_best_model {metric_for_best_model}\
        --prediction_loss_only {prediction_loss_only}\
        --save_total_limit  {save_total_limit}\
        --ignore_data_skip {ignore_data_skip} \
        --train_file  {train_file} \
        --do_train True \
        --output_dir  {output_dir}"
    os.system(command)

#    python run_clm.py {args}

    print(f"Elapsed time:",elapsed_time())
    print(args)
    return "Training/adaptation finish"

#interact is used in the colab environment not in the terminal environment

#interact(training_model,
#         model_name_or_path=['datificate/gpt2-small-spanish','DeepESP/gpt2-spanish','flax-community/gpt-2-spanish','mrm8488/spanish-gpt2'],
#trained = training_model(msg=fixed("Select a model"),
#widgets.interact(training_model,model_name_or_path='/home/gerardo/Documents/GAM/GPT-2/gpt2-small-spanish', \
print("Va a correr training_model")

#trained = training_model(msg=fixed("Select a model"),\
#                         model_name_or_path='/home/gerardo/Documents/GAM/GPT-2/gpt2-small-spanish',\
#                          do_train = fixed(True), \
#                          overwrite_cache=fixed(True), \
#                          train_file='/home/gerardo/Documents/GAM/LIPN_GitLab/data/microfictions/es/parsehub/CiudadSeva/T_cs_all_train.txt',\
#                          block_size=128,\
#                         num_train_epochs=40,\
#                         adam_beta1=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.99),\
#                         adam_beta2=widgets.FloatSlider(min=0.0, max=1.0, step=0.001, value=0.999),\
#                         adam_epsilon=[1e-4,1e-3,1e-2,1e-1],\
#                         max_grad_norm=1.0,\
#                         save_total_limit=0,\
#                         overwrite_output_dir=True,\
#                         metric_for_best_model=True,\
#                         prediction_loss_only=True,\
#                         validation_file= fixed(None),\
#                         ignore_data_skip=True,\
#                         do_eval=fixed(False),\
#                         output_dir='home/gerardo/Documents/GAM/LIPN_GitLab/experiments/monterroso.v0.18oct2021/es/output/pass_1'                         )

#GPT-2 model already trained in Spanish, from Hugging Face
#trained = training_model(model_name_or_path='/home/gerardo/Documents/GAM/GPT-2/gpt2-small-spanish',\
#train_file has the file that is going to be used for training        

#                         train_file='/users/aleman/microfiction/data/microfictions/es/parsehub/CiudadSeva/T_cs_mf_txt_all_train.txt',\
#                         train_file='/users/aleman/microfiction/data/microfictions/es/parsehub/CiudadSeva/T_cs_mf_txt1_train.txt',\
#changing block_size from 128 to 64
#adding --per_device_batch_size 4



trained = training_model(model_name_or_path='/users/gerardo.aleman/microfiction/data/microfictions/es/gpt2-small-spanish',\
                         do_train = True, \
                         overwrite_cache=True, \
                         train_file='/users/gerardo.aleman/microfiction/data/microfictions/es/parsehub/CiudadSeva/T_cs_mf_txt_all_train.txt',\
                         block_size=1,\
                         per_device_eval_batch_size=1,
                         per_device_train_batch_size=1,
                         num_train_epochs=40,\
                         adam_beta1=0.99, \
                         adam_beta2=0.999,\
                         adam_epsilon=1e-1, \
                         max_grad_norm=1.0,\
                         save_total_limit=0,\
                         overwrite_output_dir=True,\
                         metric_for_best_model=True,\
                         prediction_loss_only=True,\
                         validation_file= None,\
                         ignore_data_skip=True, \
                         output_dir='/users/gerardo.aleman/microfiction/experiments/monterroso.v0.18oct2021/es/output/pass_2',\
                         do_eval=False)




#!rm -rf /root/.cache/huggingface/datasets

#Validation of the first model
for validation_file in validation_files:
  print(f"--------------> Validating: {validation_file}")
 # training_model(msg="Validatiing",
training_model(overwrite_cache=True,\
#                model_name_or_path="home/gerardo/Documents/GAM/LIPN_GitLab/experiments/monterroso.v0.18oct2021/es/output/pass_1",\
                model_name_or_path="/users/gerardo.aleman/microfiction/experiments/monterroso.v0.18oct2021/es/output/pass_2",\
                do_train = False,\
                validation_file='/users/gerardo.aleman/microfiction/data/microfictions/es/parsehub/CiudadSeva/V_cs_mf_txt_all_validate.txt',\
                do_eval=True)

#model = "/output/pass_1"
#model = "/home/gerardo/Documents/GAM/LIPN_GitLab/experiments/monterroso.v0.18oct2021/es/output/pass_1"
model = "/users/gerardo.aleman/microfiction/experiments/monterroso.v0.18oct2021/es/output/pass_2"
model_text = pipeline('text-generation',model=model, tokenizer=model,)

microficcion=model_text("~ Primer microficción~ ",max_length=300) [0]['generated_text']
microficcion=microficcion.replace(" ~< ","\n\n").replace(" -- ","\n").replace(" <- ","\n").replace("<- ","\n").replace(" -> ","\n").replace("~ ","").replace(" >< ","\n\n").replace(" <> ","\n").replace(" ---- ","\n\n").replace(" >","\n\n").replace("-->","\n\n")
print(microficcion)
