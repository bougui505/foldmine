#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################
import os
import BLASTloader
import torch
import encoder
import time
from eta import ETA
import datetime

# See: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn(batch):
    return batch


def get_batch_test():
    """
    >>> batch = get_batch_test()
    >>> len(batch)
    3
    >>> batch
    [(Data(), Data()), (Data(), Data()), (Data(edge_index=[2, 717], node_id=[154], num_nodes=154, x=[154, 20]), Data(edge_index=[2, ...], node_id=[...], num_nodes=..., x=[..., 20]))]
    """
    dataset = BLASTloader.PDBdataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4, collate_fn=collate_fn)
    for batch in dataloader:
        break
    return batch


def get_norm(out):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.ProteinGraphModel(latent_dim=512)
    >>> out = forward_batch(batch, model)
    >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in out]
    [(torch.Size([1, 512]), torch.Size([1, 512]))]
    >>> get_norm(out)
    tensor(..., grad_fn=<MeanBackward0>)
    """
    z_full_list = torch.cat([e[0] for e in out])  # torch.Size([4, 512])
    z_fragment_list = torch.cat([e[1] for e in out])  # torch.Size([4, 512])
    z = torch.cat((z_full_list, z_fragment_list))  # torch.Size([8, 512])
    norms = torch.linalg.norm(z, dim=1)
    return norms.mean()


def forward_batch(batch, model):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.ProteinGraphModel()
    >>> out = forward_batch(batch, model)
    >>> [(z_anchor.shape, z_positive.shape) for z_anchor, z_positive in out]
    [(torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    out = []
    for anchor, positive in batch:
        if anchor.x is not None and positive.x is not None:
            z_anchor = model(anchor)
            z_positive = model(positive)
            out.append((z_anchor, z_positive))
    return out


def get_contrastive_loss(out, tau=1.):
    """
    >>> n = 3
    >>> out = [(torch.randn(1, 512), torch.randn(1, 512)) for i in range(n)]
    >>> loss = get_contrastive_loss(out)
    >>> loss
    tensor(...)
    """
    n = len(out)
    z_anchor_list = [e[0] for e in out]
    z_positive_list = [e[1] for e in out]
    loss = 0.
    for i in range(n):
        z_full_i = z_anchor_list[i]
        # z_fragment_i = z_fragment_list[i]
        den = 0.
        for j in range(n):
            z_full_j = z_anchor_list[j]
            z_fragment_j = z_positive_list[j]
            if i == j:
                sim_num = torch.matmul(z_full_i, z_fragment_j.T)
                # log(f'z_full_i: {z_full_i}')
                # log(f'sim_num: {sim_num}')
                num = torch.exp(sim_num / tau)
            else:
                sim_den = torch.matmul(z_full_i, z_full_j.T)
                # log(f'sim_den: {sim_den}')
                den += torch.exp(sim_den / tau)
        # log(f'num:{num}, den: {den}')
        loss -= torch.log((num + 1e-8) / (den + 1e-8))
    if n > 0:
        loss = loss / n
    loss = torch.squeeze(loss)
    assert not torch.isnan(loss), 'loss is nan'
    return loss


def load_model(filename, latent_dim=512):
    """
    >>> model = encoder.ProteinGraphModel()
    >>> torch.save(model.state_dict(), 'models/gcn_test.pt')
    >>> gcn = load_model('models/gcn_test.pt')
    Loading GCN model
    >>> batch = get_batch_test()
    >>> out = forward_batch(batch, gcn)
    >>> [(z_anchor.shape, z_positive.shape) for z_anchor, z_positive in out]
    [(torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = encoder.ProteinGraphModel(latent_dim=latent_dim)
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    print('Loading GCN model')
    model.eval()
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def train(
        batch_size=4,
        n_epochs=20,
        latent_dim=128,
        save_each_epoch=True,
        print_each=100,
        save_each=30,  # in minutes
        modelfilename='models/sscl.pt'):
    """
    # >>> train(pdbpath='pdb', print_each=1, save_each_epoch=False, n_epochs=3, modelfilename='models/1.pt', batch_size=32)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = BLASTloader.PDBdataset()
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn)
    dataiter = iter(dataloader)
    if os.path.exists(modelfilename):
        log(f'# Loading model from {modelfilename}')
        model = load_model(filename=modelfilename, latent_dim=latent_dim)
        model.train()
    else:
        log('GCN model')
        model = encoder.ProteinGraphModel(latent_dim=latent_dim)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    t_0 = time.time()
    save_model(model, modelfilename)
    epoch = 0
    step = 0
    total_steps = n_epochs * len(dataiter)
    eta = ETA(total_steps=total_steps)
    loss = None
    while epoch < n_epochs:
        opt.zero_grad()
        try:
            step += 1
            batch = next(dataiter)
            try:
                batch = [(e[0].to(device), e[1].to(device)) for e in batch if e is not None]
            except RuntimeError:
                print('RuntimeError: CUDA out of memory. Skipping batch...')
                batch = []
            bs = len(batch)
            if bs > 0:
                out = forward_batch(batch, model)
                if len(out) > 0:
                    try:
                        loss = get_contrastive_loss(out)
                    except AssertionError:
                        print('AssertionError: loss is nan. Skipping backward...')
                    try:
                        loss.backward()
                        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                        opt.step()
                    except RuntimeError:
                        print('RuntimeError: CUDA out of memory. Skipping backward...')
            opt.zero_grad()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                save_model(model, modelfilename)
            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                if len(out) > 0:
                    norm = get_norm(out)
                if loss is not None:
                    log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|norm: {norm:.4f}|bs: {bs}|last_saved: {last_saved}| eta: {eta_val}"
                        )
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
            if save_each_epoch:
                t_0 = time.time()
                save_model(model, modelfilename)


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    import logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')

    # See: https://stackoverflow.com/a/13839732/1679629
    fileh = logging.FileHandler(logfilename, 'a')
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fileh.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(fileh)  # set the new handler
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--train', help='Train the SSCL', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model', help='Model to load or for saving', metavar='model.pt')
    parser.add_argument('--print_each', type=int, default=100)
    parser.add_argument('--latent_dim', default=512, type=int)
    parser.add_argument('--save_every',
                        help='Save the model every given number of minutes (default: 30 min)',
                        type=int,
                        default=30)
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    if args.train:
        train(n_epochs=args.epochs,
              modelfilename=args.model,
              print_each=args.print_each,
              latent_dim=args.latent_dim,
              save_each=args.save_every,
              batch_size=args.bs)
