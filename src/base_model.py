
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Activation, BatchNormalization 
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class BaseModel:

    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)
            logger.info("Loaded configuration from config.yaml")

        except Exception as e:
            raise CustomException("Error loading configuration", sys)
        

    def RecommenderNet(self, n_users, n_animes):
        try:
            embedding_size = self.config['model']['embedding_size']

            # Input layer "user"
            user = Input(name = "user", shape = [1])  
            user_embedding = Embedding(name = "user_embedding", input_dim = n_users, output_dim = embedding_size)(user)

            # Input layer "anime"
            anime = Input(name = "anime", shape = [1])  
            anime_embedding = Embedding(name = "anime_embedding", input_dim = n_animes, output_dim= embedding_size)(anime)

            # Dot layer >>> similarity between anime and user embedding
            x = Dot(name = "dot_product", normalize= True, axes = 2)([user_embedding, anime_embedding])
            x = Flatten()(x)  # flatten layer

            # Dense layer >>> flatten from multi-dim to 1 dim 
            x = Dense(1, kernel_initializer= 'he_normal')(x)
            
            # Batch layer
            x = BatchNormalization()(x)

            # Activation layer
            x = Activation("sigmoid")(x)

            # Model layer
            model = Model(inputs= [user, anime], outputs = x)
            model.compile(loss = self.config['model']['loss'], metrics = self.config['model']['metrics'], optimizer = self.config['model']['optimizer'])

            logger.info("Model created successfully!")

            return model
        
        except Exception as e:
            logger.error("Error occurred during the model architecture")
            raise CustomException("Error occured during model architecture", sys)
        
        