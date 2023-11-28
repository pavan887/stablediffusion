from flask import Flask, request, jsonify
from config import get_all_users ,load_config,add_user,delete_user
import os

HUGGINGFACE_TOKEN = "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"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "stable_diffusion_weights/pair123987"
from flask import Flask
import subprocess
app = Flask(__name__)

@app.route('/hello')
def hello_world():
    return 'Hello, Brochill'

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='ok')

# API to get all users
@app.route('/get_users', methods=['GET'])
def get_users_api():
    all_users = get_all_users()
    if all_users:
        return jsonify(all_users)
    else:
        return jsonify({"message": "No users found."})
# API to get all users
@app.route('/get_userid', methods=['GET'])
def get_userid_api():
    config = load_config()
    user_ids = list(config["users"].keys())
    if len(user_ids)>=1:
        return jsonify(user_ids)
    else:
        return jsonify({"message": "No users found."})

# API to add a user
@app.route('/add_user', methods=['POST'])
def add_user_api():
    config = load_config()
     # Get text data from form fields
    userid = request.form.get('userid')
    subject_id1 = request.form.get('subject_id1')
    subject_id2 = request.form.get('subject_id2')
    # Get file data from file input
    images1 = request.files.getlist('images1') # 'images1' should be the name of the file input in the form
    images2 = request.files.getlist('images2')# 'images2' should be the name of the file input in the form
    response = add_user(userid, subject_id1=subject_id1, images1=images1, subject_id2=subject_id2, images2=images2)
    return jsonify({"message": response})

# API to delete a user
@app.route('/delete_user', methods=['DELETE'])
def delete_user_api():
    userid = request.form.get('userid')
    response = delete_user(str(userid))
    return jsonify({"message": response})



@app.route('/train', methods=['POST'])
def sd_train():
    #data=request.get_json()
    # Check if required parameters are present in the JSON data
    #if 'pretrained_model_name_or_path' not in data:
    #    return jsonify({'error': 'Missing pretrained_model_name_or_path parameter'}), 400
    
    # Build the command to run your script
    userid = request.form.get('userid')
    config = load_config()
    #OUTPUT_DIR=usermodel_path+"/"+userid
    OUTPUT_DIR= os.path.join(str(config["usermodel_path"]),str(userid))
    save_sample_prompt = f"photo of {userid} person"
    concepts_list = config["users"][userid]
    cmd=[
            'python3',
            'train_dreambooth.py',
            f'--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"',
            f'--pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse"',
            f'--output_dir={OUTPUT_DIR}',
            f'--revision="fp16"',
            f'--with_prior_preservation',
            f'--prior_loss_weight=1.0',
            f'--seed=1337',
            f'--resolution=512',
            f'--train_batch_size=8',
            f'--train_text_encoder',
            f'--mixed_precision=fp16',
            f'--use_8bit_adam',
            f'--gradient_accumulation_steps=1',
            f'--learning_rate=1e-6 ',
            f'--lr_scheduler="constant"',
            f'--lr_warmup_steps=0 ',
            f'--num_class_images=100',
            f'--sample_batch_size=4',
            f'--max_train_steps=1500 ',
            f'--save_interval=10000 ',
            f'--save_sample_prompt={save_sample_prompt}',
            f'--concepts_list={concepts_list}'
            ]

#            f'--pretrained_model_name_or_path={data["pretrained_model_name_or_path"]}',
    ckpt=True
    # Execute the command and capture the output 
    try:
        subprocess.check_call(cmd)
        if ckpt:
            WEIGHTS_DIR=natsorted(glob(OUTPUT_DIR+ os.sep + "*"))[-1]
            ckpt_path = os.path.join(WEIGHTS_DIR , "model.ckpt")
            half_arg = ""
            #@markdown  Whether to convert to fp16, takes half the space (2GB).
            fp16 = True #@param {type: "boolean"}
            if fp16:
                half_arg = "--half"
            #!python convert_diffusers_to_original_stable_diffusion.py --model_path $WEIGHTS_DIR  --checkpoint_path $ckpt_path $half_arg
            #print(f"[*] Converted ckpt saved at {ckpt_path}")
        #print( "Training sucessfull")
        return jsonify({'message': 'Training successful'}), 200

    except subprocess.CalledProcessError as e:
        print("Training failed")
        return jsonify({'error': str(e.output)}), 500
    #return jsonify({"message": "sucess"})

@app.route('/test', methods=['POST'])
def sd_test():
    # Build the command to run your script
    userid = request.form.get('userid')
    config = load_config()
    #OUTPUT_DIR=usermodel_path+"/"+userid
    WEIGHTS_DIR = os.path.join(config['usermodel_path'],userid)
    WEIGHTS_DIR="stable_diffusion_weights/pair123987/1000"
    positive_prompt=request.form.get('pos_prompt')
    negative_prompts=request.form.get('neg_prompt')
    print(WEIGHTS_DIR)
    print(positive_prompt)
    print(negative_prompts)

    return jsonify({"message": "sucess"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9093,debug=True)
    #sd_train()


