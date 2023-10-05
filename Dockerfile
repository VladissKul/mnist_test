FROM vm-harbor-1.dh.rt.ru/datamining/vladislav.kulakov_mnist_service:latest

WORKDIR /root/work

EXPOSE 1490

COPY main.py .
COPY test_requests.py .

CMD ["python", "main.py"]