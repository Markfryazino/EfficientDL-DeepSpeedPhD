FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
WORKDIR /repos/edl_project

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3000