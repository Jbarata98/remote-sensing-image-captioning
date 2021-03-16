from configs.utils import *
from encoder_scripts.encoder_training_details import *
from encoder_scripts.train_encoder import finetune



if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'TRAIN', test = True, transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ClassificationDataset(data_folder, data_name, 'VAL', test = True, transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    model = finetune(model_type=ENCODER_MODEL, device=device)
    model = model._setup_train()

    # checkpoint =  torch.load('experiments/results/classification_finetune.pth.tar')
    checkpoint = torch.load(get_path(model = ENCODER_MODEL, is_encoder=True), map_location=torch.device("cpu"))
    print("checkpoint loaded")

    model.load_state_dict(checkpoint['model'])
    model.eval()

    def compute_acc(dataset, train_or_val):
        total_acc = torch.tensor([0.0])

        for batch, (img, target) in enumerate(dataset):

            result = model(img)
            output = torch.sigmoid(result)

            condition_1 = (output > 0.5)
            condition_2 = (target == 1)


            correct_preds = torch.sum(condition_1 * condition_2, dim=1)
            n_preds = torch.sum(condition_1, dim=1)

            acc = correct_preds.double() / n_preds
            acc[torch.isnan(acc)] = 0  # n_preds can be 0
            acc_batch = torch.mean(acc)

            total_acc += acc_batch

            if batch % 5 == 0:

                print("acc_batch", acc_batch.item())
                print("total loss", total_acc)

        print("len of train_data", len(train_loader))
        epoch_acc = (total_acc / (batch + 1)).item()
        print("epoch acc", train_or_val, epoch_acc)
        return epoch_acc

    epoch_acc_train = compute_acc(train_loader, "TRAIN")
    epoch_acc_val = compute_acc(val_loader, "VAL")

    print("train epoch", epoch_acc_train)
    print("val epoch", epoch_acc_val)