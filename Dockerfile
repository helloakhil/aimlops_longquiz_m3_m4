# pull python base image
FROM python:3.10

# copy application files
ADD /iris_classifier /iris_classifier/

# specify working directory
WORKDIR /iris_classifier

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements/requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]