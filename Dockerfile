FROM python:3.11-slim
WORKDIR /localchat
COPY ./ /localchat
RUN pip install .

EXPOSE 7860

CMD ["python", "-m", "localchat"]