
FROM python:3.9-slim
RUN pip3 install kfp
RUN pip3 install --upgrade google-cloud-aiplatform
RUN pip3 install --upgrade google-cloud-aiplatform[autologging]
RUN pip3 install --upgrade scikit-learn
RUN pip3 install --upgrade 'google-cloud-pipeline-components'


