ARG LAMBDA_TASK_ROOT="/home/app/"

FROM public.ecr.aws/c9e2p9l9/meshlib-aws-lambda:latest

COPY app.py .

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
ENTRYPOINT [ "/usr/local/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.lambda_handler" ]
WORKDIR [ "/home/app" ]