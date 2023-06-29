# Using AWS Lambda Function for mesh processing

This guide walks through creating an AWS Lambda function using a custom Docker image tailored for mesh processing.

## Prerequisites
Ensure you have the following before you begin:
- AWS account and access to AWS Management Console
- AWS Command Line Interface (CLI) installed and configured with necessary permissions
- Docker installed on your local machine

## Steps

## 1. Create Your Own App Using `examples/aws-lambda/app`

Begin by creating your own mesh processing application. You can use a sample application structure provided in `examples/aws-lambda/app`. Clone the repository or just copy the sample app, and then modify it according to your requirements. Make sure your app includes a handler function that can be invoked by AWS Lambda.

## 2. Build and Push to AWS ECR

After you have created your app, you need to package it as a Docker container and push it to Amazon Elastic Container Registry (ECR).

First, build your Docker image.
```
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

Now, create a repository in Amazon Elastic Container Registry (ECR) and push your Docker image to it.

```
aws ecr create-repository --repository-name YOUR_ECR_REPOSITORY_NAME
aws ecr get-login-password --region YOUR_AWS_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com
docker tag YOUR_DOCKER_IMAGE_NAME:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com/YOUR_ECR_REPOSITORY_NAME:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com/YOUR_ECR_REPOSITORY_NAME:latest
```

This builds the Docker image using the Dockerfile in your application directory, tags it, and pushes it to ECR.

## 3. Create a Lambda Function

With your Docker image pushed to ECR, you are now ready to create an AWS Lambda function.

- Go to the AWS Management Console and navigate to the Lambda service.
- Click on "Create function" and select "Container image" as the deployment package.
- Configure the function by giving it a name and selecting the execution role that gives it permission to execute.
- Under the "Image" section, specify the ECR image URI for the Docker image you pushed earlier.
- Adjust settings like memory and timeout according to your function's requirements.
- Click on "Create function".

Once the function is created, you can use the “Test” button in the AWS Lambda Console to invoke your function and ensure it is processing the mesh data as expected.

## 4. Monitor Your Function

Use Amazon CloudWatch to monitor your function's performance and logs. You can set up alarms and notifications based on the metrics captured by CloudWatch.

Remember to thoroughly test your function to make sure it behaves as expected.
