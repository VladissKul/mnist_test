import torch


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, num_epochs=1):
    for epoch in range(num_epochs):
        print("- " * 20)
        model.train()
        train_loss, valid_loss = 0.0, 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        print(f"[TRAIN] Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)

                loss = criterion(output, labels)
                valid_loss += loss.item() * images.size(0)
            valid_loss = valid_loss / len(test_dataloader.dataset)
            print(f"[VALID] Epoch: {epoch + 1}, Validation Loss: {valid_loss:.4f}")
