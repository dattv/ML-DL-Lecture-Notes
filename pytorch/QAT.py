import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

import QAT_utils as utils
import warnings
from tqdm import tqdm
from prettytable import PrettyTable
import onnxruntime

import argparse
import sys
import numpy as np
import os
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def prepare_model(model_name, data_dir, per_channel_quantization, bat_size_train, batch_size_test, batch_size_onnx, calibrator,
                    pretrained=True,
                  ckpt_path=None, ckpt_url=None
                  ):
    """

    Args:
        data_dir:
        per_channel_quantization:
        bat_size_train:
        batch_size_test:
        batch_size_onnx:
        calibrator:
        ckpt_path:

    Returns:

    """
    # Use 'spawn' to avoid CUDA reinitialization with forked subprocess
    torch.multiprocessing.set_start_method('spawn')

    # Initialization quantization, model and data loader
    if per_channel_quantization:
        quant_desc_input = QuantDescriptor(calib_method=calibrator)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)

    else:
        ## Force per tensor quantization for onnx runtime
        quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)

        quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_weight)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_weight)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)

    quant_modules.initialize()
    model = Net()
    print(">>>>>>>>>>>>Load checkpoint: {}".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    print(">>>>>>>>>>>>Load completed")

    quant_modules.deactivate()

    model.eval()
    model.cuda()

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)

    onnxloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return model, trainloader, testloader, onnxloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net().cuda()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# for epoch in range(20):  # loop over the dataset multiple times
#
#     net.train()
#     running_loss = 0.0
#     correct = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
#
#     net.eval()
#     cnt = 0.
#     for i, data in enumerate(testloader):
#         inputs, labels = data
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
#         cnt += len(labels)
#     correct /= cnt
#     print(f"Epoch: {epoch}, Test accuracy:  {correct:6.4f}")
#
# print('Finished Training')
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)


def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='Classification quantization flow script')

    parser.add_argument('--data-dir', '-d', type=str, help='input data folder', required=True)
    parser.add_argument('--model-name', '-m', default='resnet50', help='model name: default resnet50')
    parser.add_argument('--disable-pcq', '-dpcq', action="store_true",
                        help='disable per-channel quantization for weights')
    parser.add_argument('--out-dir', '-o', default='/tmp', help='output folder: default /tmp')
    parser.add_argument('--print_freq', '-pf', type=int, default=200, help='evaluation print frequency: default 20')
    parser.add_argument('--threshold', '-t', type=float, default=-1.0,
                        help='top1 accuracy threshold (less than 0.0 means no comparison): default -1.0')

    parser.add_argument('--batch-size-train', type=int, default=128, help='batch size for training: default 128')
    parser.add_argument('--batch-size-test', type=int, default=128, help='batch size for testing: default 128')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')

    parser.add_argument('--seed', type=int, default=12345, help='random seed: default 12345')

    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument('--ckpt-path', default='./cifar_net.pth', type=str,
                            help='path to latest checkpoint (default: none)')
    checkpoint.add_argument('--ckpt-url', default='', type=str,
                            help='url to latest checkpoint (default: none)')
    checkpoint.add_argument('--pretrained', action="store_true")

    parser.add_argument('--num-calib-batch', default=4, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')

    parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")

    parser.add_argument('--num-finetune-epochs', default=0, type=int,
                        help='Number of epochs to fine tune. 0 will disable fine tune. (default: 0)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--sensitivity', action="store_true", help="Build sensitivity profile")
    parser.add_argument('--evaluate-onnx', action="store_true", help="Evaluate exported ONNX")

    return parser

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def build_sensitivity_profile(model, criterion, data_loader_test):
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    for i, quant_layer in enumerate(quant_layer_names):
        print("Enable", quant_layer)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")
        with torch.no_grad():
            evaluate(model, criterion, data_loader_test, device="cuda")
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def export_onnx(model, onnx_filename, batch_onnx, per_channel_quantization):
    model.eval()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True # We have to shift to pytorch's fake quant ops before exporting the model to ONNX

    if per_channel_quantization:
        opset_version = 13
    else:
        opset_version = 12

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, 32, 32, device='cuda') #TODO: switch input dims by model
    try:
        torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=opset_version, do_constant_folding=True)
    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    return True

def evaluate_onnx(onnx_filename, data_loader, criterion, print_freq):
    """Evaluate accuracy on the given ONNX file using the provided data loader and criterion.
       The method returns the average top-1 accuracy on the given dataset.
    """
    print("Loading ONNX file: ", onnx_filename)
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    with torch.no_grad():
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for image, target in metric_logger.log_every(data_loader, print_freq, header):
                image = image.to("cpu", non_blocking=True)
                image_data = np.array(image)
                input_data = image_data

                # run the data through onnx runtime instead of torch model
                input_name = ort_session.get_inputs()[0].name
                raw_result = ort_session.run([], {input_name: input_data})
                output = torch.tensor((raw_result[0]))

                loss = criterion(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print('  ONNXRuntime: Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
        return metric_logger.acc1.global_avg

def main(cmd_line):
    parser = get_parser()

    args = parser.parse_args(cmd_line)
    print(parser.description)
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, data_loader_train, data_loader_test, data_loader_onnx = prepare_model(
        args.model_name,
        args.data_dir,
        not args.disable_pcq,
        args.batch_size_train,
        args.batch_size_test,
        args.batch_size_onnx,
        args.calibrator,
        args.pretrained,
        args.ckpt_path
    )

    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():
        print('Initial evaluation:')
        top1_initial = evaluate(model, criterion, data_loader_test, device="cuda", print_freq=args.print_freq)

    ## Calibrate the model
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=args.model_name,
            data_loader=data_loader_train,
            num_calib_batch=args.num_calib_batch,
            calibrator=args.calibrator,
            hist_percentile=args.percentile,
            out_dir=args.out_dir)

    ## Evaluate after calibration
    if args.num_calib_batch > 0:
        with torch.no_grad():
            print('Calibration evaluation:')
            top1_calibrated = evaluate(model, criterion, data_loader_test, device="cuda",
                                       print_freq=args.print_freq)
    else:
        top1_calibrated = -1.0

    ## Build sensitivy profile
    if args.sensitivity:
        build_sensitivity_profile(model, criterion, data_loader_test)

    ## Fineturn
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_finetune_epochs)
    for epoch in range(args.num_finetune_epochs):
        # Training a single epch
        train_one_epoch(model, criterion, optimizer, data_loader_train, "cuda", 0, args)
        lr_scheduler.step()

    if args.num_finetune_epochs > 0:
        ## Evaluate after finetuning
        with torch.no_grad():
            print('Finetune evaluation:')
            top1_finetuned = evaluate(model, criterion, data_loader_test, device="cuda")
    else:
        top1_finetuned = -1.0

    ## Export to ONNX
    onnx_filename = args.out_dir + '/' + args.model_name + ".onnx"
    top1_onnx = -1.0
    if export_onnx(model, onnx_filename, args.batch_size_onnx, not args.disable_pcq) and args.evaluate_onnx:
        ## Validate ONNX and evaluate
        top1_onnx = evaluate_onnx(onnx_filename, data_loader_onnx, criterion, args.print_freq)

    ## Print summary
    print("Accuracy summary:")
    table = PrettyTable(['Stage', 'Top1'])
    table.align['Stage'] = "l"
    table.add_row(['Initial', "{:.2f}".format(top1_initial)])
    table.add_row(['Calibrated', "{:.2f}".format(top1_calibrated)])
    table.add_row(['Finetuned', "{:.2f}".format(top1_finetuned)])
    table.add_row(['ONNX', "{:.2f}".format(top1_onnx)])
    print(table)

    ## Compare results
    if args.threshold >= 0.0:
        if args.evaluate_onnx and top1_onnx < 0.0:
            print("Failed to export/evaluate ONNX!")
            return 1
        if args.num_finetune_epochs > 0:
            if top1_finetuned >= (top1_onnx - args.threshold):
                print("Accuracy threshold was met!")
            else:
                print("Accuracy threshold was missed!")
                return 1

    return 0

if __name__ == '__main__':
    print(sys.argv[1:])

    res = main(sys.argv[1:])
    exit(res)
