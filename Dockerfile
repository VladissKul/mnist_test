FROM vm-harbor-1.dh.rt.ru/datamining/vladislav.kulakov_mnist_service:latest

WORKDIR /root/work

EXPOSE 1490

COPY index.py .
COPY testing/test_requests.py .

CMD ["python", "index.py"]