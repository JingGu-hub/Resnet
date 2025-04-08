import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import argparse

from models.Resnets import Resnets
from utils.dataset_utils import build_dataset
from utils.utils import WarmUpLR, create_file, set_seed, create_dir


def train(args, epoch, train_loader, model, optimizer, warmup_scheduler, loss_function):
    model.train()
    train_loss, train_acc = 0, 0
    for i, (images, labels, index) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        train_acc += preds.eq(labels).sum().item()

        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

    return model, train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

def evaluate(test_loader, model, loss_function):
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for i, (images, labels, index) in enumerate(test_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

def main():
    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('-seed', type=int, default=42, help='random seed')
    parser.add_argument('-gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('-model', type=str, default='resnet18', help='model type')
    parser.add_argument('-dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./data/', help='dataset directory')
    parser.add_argument('--result_dir', type=str, default='./output/results/', help='output directory')
    parser.add_argument('--save_model_dir', type=str, default='./output/models/', help='save model directory')

    # training setup
    parser.add_argument('-epochs', type=int, default=200, help='batch size for dataloader')
    parser.add_argument('-warm_epoch', type=int, default=1, help='warm up training phase')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    # other setup
    parser.add_argument('-milestones', type=list, default=[60, 120, 160], help='milestones')
    args = parser.parse_args()

    set_seed(args)
    torch.cuda.set_device(args.gpu_id)

    # get dataset
    train_dataset, test_dataset, input_channel, num_classes = build_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # define model
    model = Resnets(args.model, input_channel=input_channel, num_classes=num_classes).cuda()

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)  # learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm_epoch)

    # create file
    out_path = args.result_dir + args.dataset + '/'
    out_file = create_file(out_path, '%s_%s.txt' % (args.model, args.dataset),'epoch,train loss,train acc,test loss,test acc')
    out_total_file = create_file(out_path, 'total.txt', 'statement,test loss,test acc', exist_create_flag=False)
    create_dir(args.save_model_dir)

    # start train
    best_acc = 0.0
    last_five_accs, last_five_losses = [], []
    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm_epoch:
            train_scheduler.step(epoch)

        # train
        model, train_loss, train_acc = train(args, epoch, train_loader, model, optimizer, warmup_scheduler, loss_function)
        # test
        test_loss, test_acc = evaluate(test_loader, model, loss_function)

        print('Epoch:[%d/%d] train_loss:%f, train_acc:%f, test_loss:%f, test_acc:%f' % (epoch, args.epochs, train_loss, train_acc, test_loss, test_acc))
        with open(out_file, "a") as myfile:
            myfile.write('%d,%.4f,%.4f,%.4f,%.4f' % (epoch, train_loss, train_acc, test_loss, test_acc) + '\n')

        if (epoch + 5) >= args.epochs:
            last_five_accs.append(test_acc)
            last_five_losses.append(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_model_dir + '%s_%s_best.pth' % (args.model, args.dataset))

    # compute average accuracy and loss
    test_accuracy = round(np.mean(last_five_accs), 4)
    test_loss = round(np.mean(last_five_losses), 4)
    print('test loss:', test_loss, ', test accuracy:', test_accuracy, ', best accuracy:', best_acc)
    with open(out_total_file, "a") as myfile:
        myfile.write('%s,%s,%s,%s' % (args.model, str(test_loss), str(test_accuracy), str(best_acc)) + '\n')

if __name__ == '__main__':
    main()
