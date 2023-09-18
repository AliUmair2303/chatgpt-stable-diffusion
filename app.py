from flask import Flask, request, jsonify, send_file, Response
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import base64

app = Flask(__name__)

stability_api = client.StabilityInference(
    key='sk-jBoX7Ac0YWOOKZIlejh0e724iK1qfXd5B0oMTGdoQimAjN1i',  
    verbose=True,  
    engine="stable-diffusion-xl-beta-v2-2-2"  
)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')
    seed = data.get('seed' , None)
    steps = data.get('steps' , 30)
    cfg_scale = data.get('cfg_scale' , 8.0)
    width = data.get('width' , 512)  
    height = data.get('height' , 512)  
    samples = data.get('samples' , 1)
    
    answers = stability_api.generate (
    prompt = prompt,
    seed = seed ,
    steps = steps,
    cfg_scale = cfg_scale ,
    width = width ,  
    height = height ,   
    samples = samples ,
    sampler=generation.SAMPLER_K_DPMPP_2M
    
    )
    
    generated_images = []
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                encoded_image = base64.b64encode(artifact.binary).decode('utf-8')
                generated_images.append(encoded_image)
                
    return jsonify(images=generated_images)

if __name__ == '__main__':
    app.run(debug=True, host= '127.0.0.1' , port=5000)