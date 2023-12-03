import ml_collections
from ml_collections.config_dict import config_dict

def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = 'DrQLearner'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.encoder = 'mobilenet' # using mobilenet v3 as visual backbone

    # the following only applies when encoder is 'd4pg'
    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = 'VALID'
    config.latent_dim = 64
    # the above only applies when encoder is 'd4pg'

    config.discount = 0.99

    # rlpd configs, using an ensamble of 10 critics with 2 min qs from REDQ
    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True
    config.backup_entropy = False

    config.tau = 0.005
    config.init_temperature = 0.1
    config.target_entropy = config_dict.placeholder(float)

    return config
