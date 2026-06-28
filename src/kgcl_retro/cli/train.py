import argparse  # Explanation: imports argparse for train KGCL with edit and contrastive losses
import os  # Explanation: imports os for train KGCL with edit and contrastive losses
import sys  # Explanation: imports sys for train KGCL with edit and contrastive losses
import random  # Explanation: imports random for train KGCL with edit and contrastive losses

import numpy as np  # Explanation: imports numpy as np for train KGCL with edit and contrastive losses
from kgcl_retro.losses import ADNCE, adnce  # Explanation: imports packaged contrastive loss implementations.

import joblib  # Explanation: imports joblib for train KGCL with edit and contrastive losses
from datetime import datetime as dt  # Explanation: imports selected names needed to train KGCL with edit and contrastive losses

import torch  # Explanation: imports torch for train KGCL with edit and contrastive losses
import torch.nn as nn  # Explanation: imports torch.nn as nn for train KGCL with edit and contrastive losses
from rdkit import RDLogger  # Explanation: imports selected names needed to train KGCL with edit and contrastive losses
from torch.optim import Adam, lr_scheduler  # Explanation: imports selected names needed to train KGCL with edit and contrastive losses

from kgcl_retro.models import KGCL  # Explanation: imports the packaged KGCL model class.
from kgcl_retro.models.utils import CSVLogger, get_seq_edit_accuracy  # Explanation: imports packaged training logging and sequence accuracy helpers.
from kgcl_retro.data.datasets import RetroEditDataset, RetroEvalDataset  # Explanation: imports packaged dataset loaders.
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM  # Explanation: imports packaged feature dimensions for model configuration.
from kgcl_retro.chemistry.graphs import Vocab  # Explanation: imports the packaged vocabulary helper for edit labels.
from kgcl_retro.config.schema import (
    CONTEXTUAL_CONFIG_DEFAULTS,
    add_contextual_model_args,
    apply_env_overrides,
    apply_model_variant_defaults,
)
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.

lg = RDLogger.logger()  # Explanation: assigns an intermediate value used by later computation
lg.setLevel(RDLogger.CRITICAL)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses

DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')  # Explanation: defines a module-level constant used by the pipeline
DEFAULT_ROOT_DIR = "."  # Explanation: sets the default root containing data and experiments.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Explanation: defines a module-level constant used by the pipeline

loss_adnce_1 = ADNCE(temperature=0.3, reduction='mean', mu=0.5, sigma=1.0)  # Explanation: assigns an intermediate value used by later computation
loss_adnce_2 = ADNCE(temperature=0.3, reduction='mean', mu=0.3, sigma=1.0)  # Explanation: assigns an intermediate value used by later computation
def build_model_config(args):  # Explanation: defines build_model_config, which builds the KGCL model configuration
    model_config = {}  # Explanation: assigns an intermediate value used by later computation
    if args.get('use_rxn_class', False):  # Explanation: checks this condition to choose the next execution path
        atom_fdim = ATOM_FDIM + 10  # Explanation: computes an intermediate value for molecular graph editing
    else:  # Explanation: handles the fallback branch for the preceding condition
        atom_fdim = ATOM_FDIM  # Explanation: computes an intermediate value for molecular graph editing
    model_config['n_atom_feat'] = atom_fdim  # Explanation: assigns an intermediate value used by later computation
    if args.get('atom_message', False):  # Explanation: checks this condition to choose the next execution path
        model_config['n_bond_feat'] = BOND_FDIM  # Explanation: assigns an intermediate value used by later computation
    else:  # Explanation: handles the fallback branch for the preceding condition
        model_config['n_bond_feat'] = atom_fdim + BOND_FDIM  # Explanation: assigns an intermediate value used by later computation
    model_config['mpn_size'] = args['mpn_size']  # Explanation: assigns an intermediate value used by later computation
    model_config['mlp_size'] = args['mlp_size']  # Explanation: assigns an intermediate value used by later computation
    model_config['depth'] = args['depth']  # Explanation: assigns an intermediate value used by later computation
    model_config['dropout_mlp'] = args['dropout_mlp']  # Explanation: assigns an intermediate value used by later computation
    model_config['dropout_mpn'] = args['dropout_mpn']  # Explanation: assigns an intermediate value used by later computation
    model_config['atom_message'] = args['atom_message']  # Explanation: assigns an intermediate value used by later computation
    model_config['use_attn'] = args['use_attn']  # Explanation: assigns an intermediate value used by later computation
    model_config['n_heads'] = args['n_heads']  # Explanation: assigns an intermediate value used by later computation
    for key, default in CONTEXTUAL_CONFIG_DEFAULTS.items():
        model_config[key] = args.get(key, default)

    return apply_env_overrides(apply_model_variant_defaults(model_config))  # Explanation: returns this computed result to the caller


def save_checkpoint(model, path, epoch):  # Explanation: defines save_checkpoint, which saves model weights and reload metadata
    save_dict = {'state': model.state_dict()}  # Explanation: assigns an intermediate value used by later computation
    if hasattr(model, 'get_saveables'):  # Explanation: checks this condition to choose the next execution path
        save_dict['saveables'] = model.get_saveables()  # Explanation: assigns an intermediate value used by later computation

    name = f'epoch_{epoch + 1}.pt'  # Explanation: assigns an intermediate value used by later computation
    save_file = os.path.join(path, name)  # Explanation: builds a filesystem path
    torch.save(save_dict, save_file)  # Explanation: saves tensor batches or checkpoints

def train_epoch(args, epoch, model, train_data, loss_fn, optimizer):  # Explanation: defines train_epoch, which runs one training epoch with edit and contrastive loss
    torch.cuda.empty_cache()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    model.train()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    train_loss = 0  # Explanation: assigns an intermediate value used by later computation
    train_acc = 0  # Explanation: assigns an intermediate value used by later computation

    for batch_id, batch_data in enumerate(train_data):  # Explanation: iterates over this collection to process each item
        graph_seq_tensors, seq_labels, seq_mask = batch_data  # Explanation: computes an intermediate value for molecular graph editing
        seq_mask = seq_mask.to(DEVICE)  # Explanation: computes an intermediate value for molecular graph editing
        is_contextual = args.get("model_variant", "kgcl") != "kgcl"
        if is_contextual:
            seq_edit_scores, batch_graph_outs = model(graph_seq_tensors, seq_labels)
        else:
            seq_edit_scores, batch_graph_outs = model(graph_seq_tensors)  # Explanation: computes an intermediate value for molecular graph editing
        max_seq_len, batch_size = seq_mask.size()  # Explanation: assigns an intermediate value used by later computation
        seq_loss = []  # Explanation: computes an intermediate value for molecular graph editing
        contextual_target_indices = []

        for idx in range(max_seq_len):  # Explanation: iterates over this collection to process each item
            if is_contextual:
                target_indices = model.map_contextual_targets(seq_labels[idx], graph_seq_tensors[idx])
                contextual_target_indices.append(target_indices)
            else:
                edit_labels_idx = model.to_device(seq_labels[idx])  # Explanation: computes an intermediate value for molecular graph editing
                target_indices = torch.stack(
                    [torch.argmax(edit_labels_idx[i]).long() for i in range(batch_size)],
                    dim=0,
                )

            loss_batch = [
                seq_mask[idx][i]
                * loss_fn(
                    seq_edit_scores[idx][i].unsqueeze(0),
                    target_indices[i].unsqueeze(0).long(),
                ).sum()
                for i in range(batch_size)
            ]  # Explanation: iterates over this collection to process each item

            loss = torch.stack(loss_batch, dim=0).mean()  # Explanation: stacks tensors along a new dimension
            seq_loss.append(loss)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses

        p_feature_list = [element for element in batch_graph_outs[0]]  # Explanation: assigns an intermediate value used by later computation

        r_feature_list = []  # Explanation: assigns an intermediate value used by later computation
        seq_mask_list = seq_mask.tolist()  # Explanation: computes an intermediate value for molecular graph editing
        for col_idx in range(batch_size):  # Explanation: iterates over this collection to process each item
            first_element = batch_graph_outs[0][col_idx]  # Explanation: assigns an intermediate value used by later computation
            non_zero_elements = [  # Explanation: assigns an intermediate value used by later computation
                row[col_idx] for row_idx, row in enumerate(batch_graph_outs)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
                if not torch.equal(row[col_idx], first_element) and seq_mask_list[row_idx][col_idx]  # Explanation: checks this condition to choose the next execution path
            ]  # Explanation: closes the current multi-line expression
            if non_zero_elements:  # Explanation: checks this condition to choose the next execution path
                random_element = random.choice(non_zero_elements)  # Explanation: assigns an intermediate value used by later computation
                r_feature_list.append(random_element)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
            else:  # Explanation: handles the fallback branch for the preceding condition
                r_feature_list.append(first_element)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses


        p_features = torch.stack(p_feature_list)  # Explanation: stacks tensors along a new dimension
        r_features = torch.stack(r_feature_list)  # Explanation: stacks tensors along a new dimension

        contrastive_weight = args.get('contrastive_loss_weight')
        if contrastive_weight is None and args.get('use_rxn_class', False):  # use rxn class  # Explanation: checks this condition to choose the next execution path
            loss_c = loss_adnce_2(p_features, r_features)  # Explanation: assigns an intermediate value used by later computation
            contrastive_weight = 0.4
        elif contrastive_weight is None:  # without rxn_class  # Explanation: handles the fallback branch for the preceding condition
            loss_c = loss_adnce_1(p_features, r_features)  # Explanation: assigns an intermediate value used by later computation
            contrastive_weight = 0.3
        else:
            loss_c = loss_adnce_2(p_features, r_features) if args.get('use_rxn_class', False) else loss_adnce_1(p_features, r_features)
        total_loss = torch.stack(seq_loss).mean() + contrastive_weight * loss_c  # Explanation: stacks tensors along a new dimension
        if is_contextual:
            proposal_losses = [loss for loss in getattr(model, "last_contextual_aux_losses", []) if loss is not None]
            if proposal_losses:
                total_loss = total_loss + args.get("pair_proposal_loss_weight", 0.1) * torch.stack(proposal_losses).mean()

        if is_contextual:
            correct = 0.0
            total = 0.0
            for step_idx, target_indices in enumerate(contextual_target_indices):
                for sample_idx in range(batch_size):
                    if int(seq_mask[step_idx][sample_idx].item()) == 0:
                        continue
                    pred_idx = torch.argmax(seq_edit_scores[step_idx][sample_idx]).item()
                    correct += float(pred_idx == int(target_indices[sample_idx].item()))
                    total += 1.0
            accuracy = correct / total if total else 0.0
        else:
            accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)  # Explanation: assigns an intermediate value used by later computation
        train_loss += total_loss.item()  # Explanation: assigns an intermediate value used by later computation
        train_acc += accuracy  # Explanation: assigns an intermediate value used by later computation

        optimizer.zero_grad()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
        total_loss.backward()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
        optimizer.step()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses

        if (batch_id + 1) % args['print_every'] == 0:  # Explanation: checks this condition to choose the next execution path
            print('\repoch %d/%d, batch %d/%d, loss: %.4f, accuracy: %.4f' % (  # Explanation: prints progress or diagnostic information
            epoch + 1, args['epochs'], batch_id + 1, len(  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
                train_data), train_loss / (batch_id + 1), train_acc / (batch_id + 1)), end='', flush=True)  # Explanation: assigns an intermediate value used by later computation

    train_loss = float('%.4f' % (train_loss / len(train_data)))  # Explanation: assigns an intermediate value used by later computation
    train_acc = float('%.4f' % (train_acc / len(train_data)))  # Explanation: assigns an intermediate value used by later computation
    print('\nepoch %d/%d, train loss: %.4f, train accuracy: %.4f' %  # Explanation: prints progress or diagnostic information
          (epoch + 1, args['epochs'], train_loss, train_acc))  # Explanation: continues a structured literal or expression

    return train_loss, train_acc  # Explanation: returns this computed result to the caller


def test(model, valid_data):  # Explanation: defines test, which validates predicted edit sequences against labels
    model.eval()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    total_accuracy = 0.0  # Explanation: assigns an intermediate value used by later computation
    first_step_accuracy = 0.0  # Explanation: assigns an intermediate value used by later computation
    with torch.no_grad():  # Explanation: opens a managed resource and closes it automatically
        for batch_id, batch_data in enumerate(valid_data):  # Explanation: iterates over this collection to process each item
            prod_smi_batch, edits_batch, edits_atom_batch, rxn_classes = batch_data  # Explanation: computes an intermediate value for molecular graph editing
            for idx, prod_smi in enumerate(prod_smi_batch):  # Explanation: iterates over this collection to process each item
                if rxn_classes is None:  # Explanation: checks this condition to choose the next execution path
                    edits, edits_atom = model.predict(prod_smi)  # Explanation: assigns an intermediate value used by later computation
                else:  # Explanation: handles the fallback branch for the preceding condition
                    edits, edits_atom = model.predict(  # Explanation: assigns an intermediate value used by later computation
                        prod_smi, rxn_class=rxn_classes[idx])  # Explanation: computes an intermediate value for molecular graph editing
                if edits == edits_batch[idx] and edits_atom == edits_atom_batch[idx]:  # Explanation: checks this condition to choose the next execution path
                    total_accuracy += 1.0  # Explanation: assigns an intermediate value used by later computation
                if edits[0] == edits_batch[idx][0] and edits_atom[0] == edits_atom_batch[idx][0]:  # Explanation: checks this condition to choose the next execution path
                    first_step_accuracy += 1.0  # Explanation: assigns an intermediate value used by later computation
    valid_acc = float('%.4f' % (total_accuracy / len(valid_data)))  # Explanation: assigns an intermediate value used by later computation
    valid_first_step_acc = float(  # Explanation: assigns an intermediate value used by later computation
        '%.4f' % (first_step_accuracy / len(valid_data)))  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses

    return valid_acc, valid_first_step_acc  # Explanation: returns this computed result to the caller


def main(args):  # Explanation: defines main, which runs this script from command-line arguments

    args.update(apply_model_variant_defaults(args))
    paths = resolve_project_paths(args.get('root_dir', DEFAULT_ROOT_DIR))  # Explanation: resolves the root directory used by package CLI file operations.

    if args['dataset'] == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
        args['lr'] = 0.001  # Explanation: assigns an intermediate value used by later computation
    elif args['dataset'] == 'uspto_full':  # Explanation: checks an alternate condition after the previous branch failed
        args['lr'] = 0.0001  # Explanation: assigns an intermediate value used by later computation

    if args.get('use_rxn_class', False):  # Explanation: checks this condition to choose the next execution path
        out_dir = os.path.join(str(paths.experiments_dir),  # Explanation: builds a filesystem path
                               args['dataset'], 'with_rxn_class', DATE_TIME)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    else:  # Explanation: handles the fallback branch for the preceding condition
        out_dir = os.path.join(str(paths.experiments_dir),  # Explanation: builds a filesystem path
                               args['dataset'], 'without_rxn_class', DATE_TIME)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    os.makedirs(out_dir, exist_ok=True)  # Explanation: creates output directories when needed

    logs_filename = os.path.join(out_dir, 'logs.csv')  # Explanation: builds a filesystem path
    csv_logger = CSVLogger(  # Explanation: assigns an intermediate value used by later computation
        args=args,  # Explanation: assigns an intermediate value used by later computation
        fieldnames=['epoch', 'train_acc', 'valid_acc',  # Explanation: assigns an intermediate value used by later computation
                    'valid_first_step_acc', 'train_loss'],  # Explanation: continues the current multi-line argument or data structure
        filename=logs_filename,  # Explanation: assigns an intermediate value used by later computation
    )  # Explanation: closes the current multi-line expression

    data_dir = str(paths.dataset_dir(args['dataset']))  # Explanation: builds the selected dataset directory from the resolved root.
    # load bond, atom and lg vocab
    bond_vocab_file = os.path.join(data_dir, 'train', 'bond_vocab.txt')  # Explanation: builds a filesystem path
    atom_vocab_file = os.path.join(data_dir, 'train', 'atom_lg_vocab.txt')  # Explanation: builds a filesystem path
    bond_vocab = Vocab(joblib.load(bond_vocab_file))  # Explanation: loads processed dataset or vocabulary objects
    atom_vocab = Vocab(joblib.load(atom_vocab_file))  # Explanation: loads processed dataset or vocabulary objects

    if args.get('use_rxn_class', False):  # Explanation: checks this condition to choose the next execution path
        train_dir = os.path.join(data_dir, 'train', 'with_rxn_class')  # Explanation: builds a filesystem path
    else:  # Explanation: handles the fallback branch for the preceding condition
        train_dir = os.path.join(data_dir, 'train', 'without_rxn_class')  # Explanation: builds a filesystem path
    if args.get("model_variant", "kgcl") != "kgcl":
        train_dir = os.path.join(train_dir, args["model_variant"])
    eval_dir = os.path.join(data_dir, 'valid')  # Explanation: builds a filesystem path

    train_dataset = RetroEditDataset(data_dir=train_dir, model_variant=args.get("model_variant", "kgcl"))  # Explanation: assigns an intermediate value used by later computation
    train_data = train_dataset.loader(  # Explanation: assigns an intermediate value used by later computation
        batch_size=1, num_workers=args['num_workers'], shuffle=True)  # Explanation: assigns an intermediate value used by later computation

    valid_dataset = RetroEvalDataset(  # Explanation: assigns an intermediate value used by later computation
        data_dir=eval_dir, data_file='valid.file.kekulized', use_rxn_class=args['use_rxn_class'])  # Explanation: assigns an intermediate value used by later computation
    valid_data = valid_dataset.loader(  # Explanation: assigns an intermediate value used by later computation
        batch_size=1, num_workers=args['num_workers'])  # Explanation: assigns an intermediate value used by later computation

    model_config = build_model_config(args)  # Explanation: assigns an intermediate value used by later computation

    model = KGCL(config=model_config, atom_vocab=atom_vocab,  # Explanation: assigns an intermediate value used by later computation
                        bond_vocab=bond_vocab, device=DEVICE)  # Explanation: computes an intermediate value for molecular graph editing
    print(f'Converting model to device: {DEVICE}')  # Explanation: prints progress or diagnostic information
    sys.stdout.flush()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    model.to(DEVICE)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    print("Param Count: ", sum([x.nelement()  # Explanation: prints progress or diagnostic information
                                for x in model.parameters()]) / 10 ** 6, "M")  # Explanation: iterates over this collection to process each item
    print()  # Explanation: prints progress or diagnostic information

    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Explanation: assigns an intermediate value used by later computation
    optimizer = Adam(model.parameters(), lr=args['lr'])  # Explanation: assigns an intermediate value used by later computation
    scheduler = lr_scheduler.ReduceLROnPlateau(  # Explanation: assigns an intermediate value used by later computation
        optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'],  # Explanation: assigns an intermediate value used by later computation
        threshold_mode='abs')  # Explanation: assigns an intermediate value used by later computation

    best_acc = 0  # Explanation: assigns an intermediate value used by later computation
    for epoch in range(args['epochs']):  # Explanation: iterates over this collection to process each item

        train_loss, train_acc = train_epoch(  # Explanation: assigns an intermediate value used by later computation
            args, epoch, model, train_data, loss_fn, optimizer)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
        valid_acc, valid_first_step_acc = test(model, valid_data)  # Explanation: assigns an intermediate value used by later computation
        scheduler.step(valid_acc)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
        print('epoch %d/%d, validation accuracy: %.4f, validation_first_acc: %.4f' %  # Explanation: prints progress or diagnostic information
              (epoch + 1, args['epochs'], valid_acc, valid_first_step_acc))  # Explanation: continues a structured literal or expression
        print('---------------------------------------------------------')  # Explanation: prints progress or diagnostic information
        print()  # Explanation: prints progress or diagnostic information

        row = {  # Explanation: assigns an intermediate value used by later computation
            'epoch': str(epoch + 1),  # Explanation: continues the current multi-line argument or data structure
            'train_acc': str(train_acc),  # Explanation: continues the current multi-line argument or data structure
            'valid_acc': str(valid_acc),  # Explanation: continues the current multi-line argument or data structure
            'valid_first_step_acc': str(valid_first_step_acc),  # Explanation: continues the current multi-line argument or data structure
            'train_loss': str(train_loss),  # Explanation: continues the current multi-line argument or data structure
        }  # Explanation: closes the current multi-line expression
        csv_logger.writerow(row)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses

        # update the best accuracy for saving checkpoints
        if valid_acc >= best_acc:  # Explanation: checks this condition to choose the next execution path
            print(  # Explanation: prints progress or diagnostic information
                f'Best eval accuracy so far. Saving best model from epoch {epoch + 1} (acc={valid_acc})')  # Explanation: assigns an intermediate value used by later computation
            print('---------------------------------------------------------')  # Explanation: prints progress or diagnostic information
            print()  # Explanation: prints progress or diagnostic information
            save_checkpoint(model, out_dir, epoch)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
            best_acc = valid_acc  # Explanation: assigns an intermediate value used by later computation

    csv_logger.close()  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses
    print('Experiment finished!')  # Explanation: prints progress or diagnostic information


def build_arg_parser():  # Explanation: constructs the training argument parser for scripts and console entry points.
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='uspto_50k',  # Explanation: chooses which USPTO dataset split to use.
                        help='dataset: uspto_50k or uspto_full or uspto_mit')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--use_rxn_class', default=False,  # Explanation: enables reaction-class conditioning
                        action='store_true', help='Whether to use rxn_class')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--atom_message', default=False, action='store_true',  # Explanation: toggles atom-level versus bond-level message passing option
                        help='Node-level or Bond-level message passing')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--use_attn', default=False,  # Explanation: enables optional global atom self-attention
                        action='store_true', help='Whether to use global attention')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--n_heads', type=int, default=8,  # Explanation: sets the number of attention heads
                        help='Number of heads in Multihead attention')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--epochs', type=int, default=200,  # Explanation: sets the number of training epochs
                        help='Maximum number of epochs for training')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--mpn_size', type=int,  # Explanation: sets graph encoder hidden size
                        default=256, help='MPN hidden_dim')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--depth', type=int, default=10,  # Explanation: sets message passing depth
                        help='Number of iterations')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--dropout_mpn', type=float,  # Explanation: sets encoder dropout
                        default=0.15, help='MPN dropout rate')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--mlp_size', type=int,  # Explanation: sets edit-predictor MLP hidden size
                        default=512, help='MLP hidden_dim')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--dropout_mlp', type=float,  # Explanation: sets MLP dropout
                        default=0.2, help='MLP dropout rate')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # Explanation: sets learning rate

    parser.add_argument('--patience', type=int, default=5,  # Explanation: sets scheduler patience
                        help='Number of epochs with no improvement after which lr will be reduced')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--factor', type=float, default=0.8,  # Explanation: sets learning-rate decay factor
                        help='Factor by which the lr will be reduced')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--thresh', type=float, default=0.01,  # Explanation: sets validation improvement threshold
                        help='Threshold for measuring the new optimum')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--max_clip', type=int, default=10,  # Explanation: sets gradient clipping limit
                        help='Maximum number of gradient clip')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--print_every', type=int,  # Explanation: sets logging frequency
                        default=200, help='Print during train process')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--num_workers', default=24,  # Explanation: sets dataloader worker count
                        help='Number of processes for data loading')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR,  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.
    add_contextual_model_args(parser)
    return parser  # Explanation: returns the configured parser to the caller.


def cli_main():  # Explanation: parses command-line arguments and launches KGCL training.
    parser = build_arg_parser()  # Explanation: creates the shared training argument parser.
    args = parser.parse_args().__dict__  # Explanation: parses command-line options into the dict expected by main.
    args = apply_model_variant_defaults(args)
    main(args)  # Explanation: executes this statement as part of train KGCL with edit and contrastive losses


if __name__ == '__main__':  # Explanation: runs the CLI entry point only when this file is executed directly
    cli_main()  # Explanation: delegates script execution to the reusable console entry point.
