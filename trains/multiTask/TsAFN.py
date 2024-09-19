import os
import logging
import pickle as plk
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from torch import optim, Tensor
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = logging.getLogger('MSA')

class TsAFN():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.df = pd.DataFrame(columns=['fusion', 'text', 'audio', 'vision'])

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }

        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:   # 'MTAV'
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += self.weighted_loss(outputs[m], outputs['Feature_'+m], self.label_map[self.name_map[m]][indexes],
                                                   indexes=indexes, mode=self.name_map[m])
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_M'].detach()
                    f_text = {'text': outputs['Feature_T'].detach()}
                    f_audio = {'audio': outputs['Feature_A'].detach()}
                    f_vision = {'vision': outputs['Feature_V'].detach()}
                    if epochs > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)
                        # if epochs%4==0:
                        # self.plot_pos_neg_comparison(self.label_map, epochs)
                    self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

                row = [torch.tensor(self.label_map[key][100]).detach().cpu() for key in self.label_map.keys()]
                self.df = self.df.append(pd.Series(row, index=self.df.columns), ignore_index=True)

            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName,
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                self.plot_emotion_polarity_per_epoch(self.df, epochs)

                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.weighted_loss(outputs['M'], outputs['Feature_M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results

    def weighted_loss(self, y_pred, feature, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    
    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text['text']
        self.feature_map['audio'][indexes] = f_audio['audio']
        self.feature_map['vision'][indexes] = f_vision['vision']
    
    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8
        def update_single_label(f_fusion, f_single, f_s1, f_s2, mode):
            d_fs = torch.norm(list(f_single.values())[0] - f_fusion, dim=-1)
            d_fs1 = torch.norm(list(f_s1.values())[0] - f_fusion, dim=-1)
            d_fs2 = torch.norm(list(f_s2.values())[0] - f_fusion, dim=-1)

            d_ss1 = torch.norm(list(f_single.values())[0] - list(f_s1.values())[0])
            d_ss2 = torch.norm(list(f_single.values())[0] - list(f_s2.values())[0])

            beta_ss1 = (d_fs - d_fs1) / (d_ss1 + MIN)
            beta_ss2 = (d_fs - d_fs2) / (d_ss2 + MIN)
            beta_s1s2 = (d_fs1 - d_fs2) / (d_ss2 + MIN)

            alpha_s = (beta_ss1 + beta_ss2) / beta_ss1
            alpha_s1 = (beta_ss1 + beta_s1s2) / beta_s1s2
            alpha_s2 = (beta_ss2 + beta_s1s2) / beta_ss2

            key_s1 = list(f_s1.keys())[0]
            key_s2 = list(f_s2.keys())[0]
            new_labels = (1/3) * (3 * self.label_map['fusion'][indexes] + alpha_s +
                                  (alpha_s / alpha_s1) * (self.label_map[key_s1][indexes] - self.label_map['fusion'][indexes])+
                                  (alpha_s / alpha_s2) * (self.label_map[key_s2][indexes] - self.label_map['fusion'][indexes]))
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        update_single_label(f_fusion, f_text, f_audio, f_vision, mode='text')
        update_single_label(f_fusion, f_audio, f_text, f_vision, mode='audio')
        update_single_label(f_fusion, f_vision, f_text, f_audio, mode='vision')

    def plot_emotion_polarity_per_epoch(self, df, epoch_count):
        data_dict = {}
        for column in df.columns:
            data_dict[column] = df[column].tolist()

        # Assuming all modalities have the same number of epochs
        epochs = [x for x in range(1, len(data_dict['fusion']) + 1)]
        markers = ['^', 's', 'o', '*']  # Triangle, Square, and Circle markers

        plt.figure(figsize=(10, 6))
        for i, (modality, emotions) in enumerate(data_dict.items()):
            marker = markers[i % len(markers)]  # Cycle through markers
            plt.plot(epochs, emotions, label=modality, marker=marker, linestyle='-')

        # for modality, emotions in df.items():
        #     plt.plot(epochs, emotions, label=modality)

        plt.xlabel('Epoch')
        plt.ylabel('Sentiment Polarity')
        plt.title('Sentiment Polarity Updated with Epoch(SIMS)')
        plt.xticks(epochs)
        plt.yticks(range(-3, 4))
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plot_pos_neg{epoch_count + 1}.png')
        plt.show()
        # plt.clf()


    def plot_pos_neg_comparison(self, label_map, epoch_count):

        data1 = {label_map[key] for key in ['text', 'audio', 'vision']}
        df1 = pd.DataFrame(data1).T

        def count_positive_negative(df):
            positive_counts = df[df >= 0].count()
            negative_counts = df[df < 0].count()
            return positive_counts, negative_counts

        dict1 = {col: count_positive_negative(df1[col]) for col in df1.columns}

        columns = df1.columns
        positive_counts_df1 = [dict1[col][0] for col in columns]
        negative_counts_df1 = [dict1[col][1] for col in columns]

        fig, ax = plt.subplots(figsize=(5, 6))
        width = 0.35
        x = range(len(columns))

        rects1 = ax.bar(x, positive_counts_df1, width, label='Positive', color='#FFB6C1')
        rects2 = ax.bar(x, negative_counts_df1, width, label='Negative', bottom=positive_counts_df1,
                        color='#1F77B4')

        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        ax.set_xlabel('Modalities')
        ax.set_ylabel('Counts')
        ax.set_title('Pos/Neg Value Counts Comparison(MOSEI)')
        ax.legend()

        plt.tight_layout()
        plt.text(x[0], -3.5, 'text', ha='center')
        plt.text(x[len(x) // 2], -3.5, 'audio', ha='center')
        plt.text(x[-1], -3.5, 'vision', ha='center')

        plt.savefig(f'plot_update{epoch_count + 1}.png')
        # plt.show()

        # plt.clf()
