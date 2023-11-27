HUGGINGFACE_TOKEN = "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"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "stable_diffusion_weights/pair123987"
from flask import Flask
import subprocess
app = Flask(__name__)

@app.route('/hello')
def hello_world():
    return 'Hello, Pavan'
#@app.route('/train')
def sd_train():
    #data=request.get_json()
    # Check if required parameters are present in the JSON data
    #if 'pretrained_model_name_or_path' not in data:
    #    return jsonify({'error': 'Missing pretrained_model_name_or_path parameter'}), 400
    
    # Build the command to run your script
    cmd=[
            'python3',
            'train_dreambooth.py',
            f'--pretrained_model_name_or_path={"runwayml/stable-diffusion-v1-5"}',
            f'--pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse"',
            f'--output_dir="sd_user_weights/kajal123987/',
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
            f'--save_sample_prompt="photo of kajal123098 with pavan123987 person"',
            f'--concepts_list=concepts_list.json'
            ]

#            f'--pretrained_model_name_or_path={data["pretrained_model_name_or_path"]}',

    # Execute the command and capture the output 
    try:
        subprocess.check_call(cmd)
        #return jsonify({'message': 'Training successful'}), 200
        print( "Training sucessfull")

    except subprocess.CalledProcessError as e:
        #return jsonify({'error': str(e.output)}), 500
        print("Training failed")

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=9093,debug=True)
    sd_train()


