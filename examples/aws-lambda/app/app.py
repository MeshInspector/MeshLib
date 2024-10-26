import boto3
import json
import sys
import requests

from meshlib import mrmeshpy


def handler(event, context):
    if 'url' not in event:
        resp_body = {'message': 'no url'}
        return {
            "statusCode": 400,
            "body": json.dumps(resp_body)
        }

    response = requests.get(event['url'])
    with open("/tmp/my.stl", "wb") as fin:
        fin.write(response.content)

    mesh = mrmeshpy.loadMesh("/tmp/my.stl")
    settings = mrmeshpy.DecimateSettings()
    settings.maxError = 0.001
    result = mrmeshpy.decimateMesh(mesh, settings)

    with open("/tmp/res.stl", "wb") as fout:
        mrmeshpy.saveMesh(mesh, "*.stl", fout)

    s3 = boto3.client('s3')
    try:
        s3.upload_file("/tmp/res.stl", 'stl-for-video', "res.stl")
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': 'stl-for-video',
                'Key': "res.stl"
            },
            ExpiresIn=3600
        )
        print("Upload Successful", url)
        resp_body = {'result': url}
        return {
            "statusCode": 200,
            "body": json.dumps(resp_body)
        }
    except Exception as e:
        resp_body = {'message': e}
        return {
            "statusCode": 500,
            "body": json.dumps(resp_body)
        }