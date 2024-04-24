import wandb

class Reporter:
    ''' Reporter class. '''

    def __init__(self, config, logger):
        self.logger = logger
        self.sample_rate = config['sample_rate']
        try:
            self.wandb_credentials = config['logs']['wandb_credentials']
        except:
            self.wandb_credentials = None
            self.logger.info('WARNING: Reporter could not read wandb_credentials,'
                             'wandb logs are turned off')
        if self.wandb_credentials is not None:
            wandb.login(key=self.wandb_credentials['wandb_key'])
            wandb.init(
                project=self.wandb_credentials['wandb_project'],
                entity=self.wandb_credentials['wandb_entity'],
                name=self.wandb_credentials['run_name'],
                config=dict(config)
            )

    def wandb_format_name(self, name):
        ''' Wandb name format. '''
        return f'{name}_{self.mode}'

    def wandb_format_number(self, number_name, number):
        ''' Wandb number format. '''
        return {
            self.wandb_format_name(number_name): number
        }

    def wandb_format_audio(self, audio):
        ''' Wandb audio format. '''
        audio = audio.detach().cpu().numpy()
        return wandb.Audio(audio, sample_rate=self.sample_rate)

    def wandb_finish(self):
        ''' Wandb run finish. '''
        wandb.finish()

    def add_and_report(self, logs=None, mode='train'):
        ''' Report update. '''
        self.mode = mode

        if self.mode == 'train':
            self.logger.info('TRAIN LOGS!')
            wandb.log(self.wandb_format_number('loss', logs['loss']), step=logs['step'])
            if logs['metrics'] is not None:
                metrics = logs['metrics']
                for metric in logs['metrics']:
                    wandb.log(self.wandb_format_number(metric, metrics[metric]), step=logs['step'])

        if self.mode == 'eval':
            self.logger.info('EVAL LOGS!')
            wandb.log(self.wandb_format_number('loss', logs['loss']), step=logs['step'])
            if logs['metrics'] is not None:
                metrics = logs['metrics']
                for metric in logs['metrics']:
                    wandb.log(self.wandb_format_number(metric, metrics[metric]), step=logs['step'])

        if self.mode == 'inference':
            self.logger.info('INFERENCE!')

            columns = ['mix_name', 'mix', 's1', 's2']
            table = wandb.Table(columns=columns)

            mixtures = logs['mixtures']

            for id in mixtures:
                mix_id = mixtures[id]

                mix = self.wandb_format_audio(mix_id['mix'])
                s1_target = self.wandb_format_audio(mix_id['s1_target'])
                s2_target = self.wandb_format_audio(mix_id['s2_target'])
                s1_est = self.wandb_format_audio(mix_id['s1_estimated'])
                s2_est = self.wandb_format_audio(mix_id['s2_estimated'])

                table.add_data(str(id) + '_target', mix, s1_target, s2_target)
                table.add_data(str(id) + '_estimated', None, s1_est, s2_est)
            
            wandb.log({'inference (using best model)': table}, step=logs['step'])
        
        if self.mode == 'inference_spe':
            self.logger.info('INFERENCE_spe!')

            columns = ['mix_name', 'mix', 'target', 'estimated', 'reference']
            table = wandb.Table(columns=columns)

            mixtures = logs['mixtures']

            for id in mixtures:
                mix_id = mixtures[id]

                mix = self.wandb_format_audio(mix_id['mix'])
                target = self.wandb_format_audio(mix_id['target'])
                estimated = self.wandb_format_audio(mix_id['estimated'])
                reference = self.wandb_format_audio(mix_id['reference'])

                table.add_data(str(id), mix, target, estimated, reference)

            wandb.log({'inference_spe (using best model)': table}, step=logs['step'])
