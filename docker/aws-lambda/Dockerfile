ARG LAMBDA_TASK_ROOT="/home/app/"

FROM python:3.8-slim-bullseye

RUN mkdir -p /home/app
WORKDIR "/home/app"
RUN chmod -R 777 "/home/app"

COPY requirements.txt  .
RUN pip3 install -r requirements.txt

# Copy function code
COPY app.py .

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
ENTRYPOINT [ "/usr/local/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.lambda_handler" ]
WORKDIR [ "/home/app" ]
