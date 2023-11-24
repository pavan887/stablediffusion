from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

CONFIG_FILE = "config.json"

def load_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        return None

def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=4)

def add_user(userid,subject_id1="" ,images1=None,subject_id2="",images2=None):
    #config = load_config()
    instance_prompt1=f"Photo of {subject_id1} person"
    instance_prompt2=f"Photo of {subject_id2} person"
    class_prompt="Photo of a person"
    instance_data_dir1=config["userdata_path"]+subject_id1
    instance_data_dir2=config["userdata_path"]+subject_id2
    os.makedirs(instance_data_dir1, exist_ok=True)
    os.makedirs(instance_data_dir2, exist_ok=True)
    # Save the images to the specified directories
    save_images(images1, os.path.join(instance_data_dir1, subject_id1))
    save_images(images2, os.path.join(instance_data_dir2, subject_id2))
    class_data_dir=config["userdata_path"]+"person"
    if config is None:
        config = {
            "userdata_path": "path/to/folder",
            "usermodel_path": "path/to/folder",
            "HUGGINGFACE_TOKEN": "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",
            "users": {}
        }

    config["users"][userid] = [

        {"instance_prompt": instance_prompt1,
        "class_prompt": class_prompt,
        "instance_data_dir": instance_data_dir1,
        "class_data_dir": class_data_dir
        },
        {"instance_prompt": instance_prompt2,
        "class_prompt": class_prompt,
        "instance_data_dir": instance_data_dir2,
        "class_data_dir": class_data_dir
        }
    ]

    save_config(config)
    print(f"User {userid} added successfully.")

def delete_user(userid):
    config = load_config()

    if config is not None and userid in config["users"]:
        user_data = config["users"][userid]

        # Delete instance data directories
        for instance in user_data:
            instance_data_dir = instance["instance_data_dir"]
            if os.path.exists(instance_data_dir):
                os.rmdir(instance_data_dir)
    
        del config["users"][userid]

        save_config(config)

        print(f"User {userid} deleted successfully.")
    else:
        print(f"User {userid} not found.")
def get_all_users():
    config = load_config()

    if config is not None and "users" in config:
        return config["users"]
    else:
        return 
def save_image(image, filepath):
    if image:
        image.save(filepath)
def save_images(images, save_path):
    os.makedirs(save_path, exist_ok=True)

    for i, image in enumerate(images):
        save_image(image, os.path.join(save_path, f"image{i+1}.jpg"))
# API to get all users
@app.route('/get_users', methods=['GET'])
def get_users_api():
    all_users = get_all_users()
    if all_users:
        return jsonify(all_users)
    else:
        return jsonify({"message": "No users found."})

# API to add a user
@app.route('/add_user', methods=['POST'])
def add_user_api():
    config = load_config()
    data = request.get_json()
    userid = data.get('userid')
    subject_id1 = data.get('subject_id1')
    subject_id2 = data.get('subject_id2')
    images1 = request.files.getlist('images1')  # 'images1' should be the name of the file input in the form
    images2 = request.files.getlist('images2')  # 'images2' should be the name of the file input in the form
    response = add_user(userid, subject_id1=subject_id1, images1=images1, subject_id2=subject_id2, images2=images2)
    return jsonify({"message": response})

# API to delete a user
@app.route('/delete_user', methods=['DELETE'])
def delete_user_api():
    userid = request.args.get('userid')
    response = delete_user(userid)
    return jsonify({"message": response})


'''

# GET api
all_users = get_all_users()
if all_users:
    print("All Users:")
    for user_id, user_data in all_users.items():
        print(f"User ID: {user_id}, Data: {user_data}")
else:
    print("No users found.")
    
# ADD api
add_user(
    "userid1",
    "deepu",

    "pavan"
    
)

# DELETE api
delete_user("userid2")

# GET api
all_users = get_all_users()
if all_users:
    print("All Users:")
    for user_id, user_data in all_users.items():
        print(f"User ID: {user_id}, Data: {user_data}")
else:
    print("No users found.")
'''
