import sys
import numpy as np
import os
import gym
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, \
                                    Activation, BatchNormalization, ReLU, LeakyReLU, ELU, Dropout, AlphaDropout, Add
from tensorflow.keras.models import Model
# to be consistent with my standard loading of the Keras backend in Jupyter notebooks:  
from tensorflow.keras import backend as B      
from tensorflow.keras.optimizers import Adam
import numpy as np
import time 
import os
import sklearn # could be used for scalers
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpat 

# tensorflow and keras 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend as B 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

finalgames = ['BattleZone','DoubleDunk','NameThisGame','Phoenix','Qbert']

#MAIN PARAMETERS
screensize = 160
z_dim = 128
INITIAL_EPOCH = 0 
n_epochs      = 150
SOLUTION_TYPE = 3
solution_type = SOLUTION_TYPE
#INPUT_DIM = (screensize,screensize,1) #if grayscale
INPUT_DIM = (screensize,screensize,3) # if rgb
loss_type = 0 #0=BCE - 1=MSE , sigmoid with 0 and linear with 1
act = 1 #0 LeakyReLu, 1 ReLU, 2= SeLU
fact = 0
use_batch_norm  = True
use_dropout     = False
dropout_rate    = 0.1
n_ch  = INPUT_DIM[2]   # number of channels
if loss_type == 0:
    fact = 5.0
else:
    fact = 2.0e-2
    
# A child class of Model() to control train_step with GradientTape() 
class VAE(keras.Model): 
    
    # We use our self defined __init__() to provide a reference MyVAE 
    # to an object of type "MyVariationalAutoencoder" 
    # This in turn allows us to address the Encoder and the Decoder  
    def __init__(self, MyVAE, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.MyVAE   = MyVAE 
        self.encoder = self.MyVAE.encoder
        self.decoder = self.MyVAE.decoder
        
        # A factor to control the ratio between the KL loss and the reconstruction loss 
        self.fact = MyVAE.fact
        
        # A counter 
        self.count = 0 
        
        # A factor to scale the absolute values of the losses 
        # e.g. by the number of pixels of an image
        self.scale_fact = 1.0  # no scaling
        # self.scale_fact = tf.constant(self.MyVAE.input_dim[0] * self.MyVAE.input_dim[1], dtype=tf.float32)
        self.f_scale    = 1. / self.scale_fact
        
        # loss type : 0: BCE, 1: MSE 
        self.loss_type = self.MyVAE.loss_type
        
        # track loss development via metrics 
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reco_loss_tracker  = keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x, z_m, z_var = self.encoder(inputs)
        return self.decoder(x)  

    # Overwrite the metrics() of Model() - use getter mechanism  
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reco_loss_tracker,
            self.kl_loss_tracker
        ]

    # Core function to control all operations regarding eager differentiation operations, 
    # i.e. the calculation of loss terms with respect to tensors and differentiation variables 
    # and metrics data 
    def train_step(self, data):
        # We use the GradientTape context to record differntiation operations/results 
        #self.count += 1 
        
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encoder(data)
            reconstruction = self.decoder(z)
            #reco_shape = tf.shape(self.reconstruction)
            #print("reco_shape = ", reco_shape, self.reconstruction.shape, data.shape)
            
            #BCE loss (Binary Cross Entropy) 
            if self.loss_type == 0: 
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                ) * self.f_scale
            
            # MSE loss (Mean Squared Error) 
            if self.loss_type == 1: 
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.mse(data, reconstruction), axis=(1, 2)
                    )
                ) * self.f_scale
            
            kl_loss = -0.5 * self.fact * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  
            total_loss = reconstruction_loss + kl_loss 
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #if self.count == 1: 
            
        self.total_loss_tracker.update_state(total_loss)
        self.reco_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reco_loss": self.reco_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def compile_VAE(self, learning_rate):

        # Optimizer
        # ~~~~~~~~~ 
        optimizer = Adam(learning_rate=learning_rate)
        # save the learning rate for possible intermediate output to files 
        self.learning_rate = learning_rate
        self.compile(optimizer=optimizer)
        

class My_KL_Layer(Layer):
    '''
    @note: Returns the input layers ! Required to allow for z-point calculation
           in a final Lambda layer of the Encoder model    
    '''
    # Standard initialization of layers 
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(My_KL_Layer, self).__init__(*args, **kwargs)

    # The implementation interface of the Layer
    def call(self, inputs, fact = 4.5e-4):
        mu      = inputs[0]
        log_var = inputs[1]
        # Note: from other analysis we know that the backend applies tf.math.functions 
        # "fact" must be adjusted - for MNIST reasonable values are in the range of 0.65e-4 to 6.5e-4
        kl_mean_batch = - fact * B.mean(1 + log_var - B.square(mu) - B.exp(log_var))
        # We add the loss via the layer's add_loss() - it will be added up to other losses of the model     
        self.add_loss(kl_mean_batch, inputs=inputs)
        # We add the loss information to the metrics displayed during training 
        self.add_metric(kl_mean_batch, name='kl', aggregation='mean')
        return inputs

        
def identity_block(x,filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,strides=1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters=filter, kernel_size=3,strides=1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    #x_skip = tf.keras.layers.Conv2D(filter, 1, strides = 1)(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block_transpose(x,filter):
        # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2DTranspose(kernel_size=3, strides=1, filters=filter, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = Conv2DTranspose(kernel_size=3, strides=1, filters=filter, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    #x_skip = tf.keras.layers.Conv2DTranspose(filter, 1, strides = 1)(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

#https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/

class CustomLambda(Layer):
    def __init__(self, *args, **kwargs):
        super(CustomLambda,self).__init__(*args,**kwargs)
        
    def call(self,args):
            '''
            A point in the latent space is calculated statistically 
            around an optimized mu for each sample 
            '''
            mu, log_var = args # Note: These are 1D tensors !
            epsilon = B.random_normal(shape=B.shape(mu), mean=0., stddev=1.)
            return mu + B.exp(log_var / 2) * epsilon
# The Main class 
# ~~~~~~~~~~~~~~
class MyVariationalAutoencoder():
    '''
    Coding suggestions of D. Foster and F. Chollet were modified and extended by RMO 
    @version: V0.1, 25.04 
    @change:  added b_build_all 
    @version: V0.2, 08.05 
    @change:  Handling of the KL-loss via functions (partially not working)  
    @version: V0.3, 29.05 
    @change:  Handling of the KL-loss function via a customized Encoder layer 
    '''
    
    def __init__(self
        , input_dim                  # the shape of the input tensors (for MNIST (28,28,1)) 
        , encoder_conv_filters       # number of maps of the different Conv2D layers   
        , encoder_conv_kernel_size   # kernel sizes of the Conv2D layers 
        , encoder_conv_strides       # strides - here also used to reduce spatial resolution avoid pooling layers 
                                     # used instead of Pooling layers 
        , encoder_conv_padding       # padding - valid or same  
        
        , decoder_conv_t_filters     # number of maps in Con2DTranspose layers 
        , decoder_conv_t_kernel_size # kernel sizes of Conv2D Transpose layers  
        , decoder_conv_t_strides     # strides for Conv2dTranspose layers - inverts spatial resolution
        , decoder_conv_t_padding     # padding - valid or same  
        
        , z_dim                      # A good start is 16 or 24  
        , solution_type  = 0         # Which type of solution for the KL loss calculation ?
        , act            = 0         # Which type of activation function?  
        , fact           = 0.65e-4   # Factor for the KL loss (0.5e-4 < fact < 1.e-3is reasonable) 
        , loss_type      = 0         # 0: BCE, 1: MSE   
        , use_batch_norm = False     # Shall BatchNormalization be used after Conv2D layers? 
        , use_dropout    = False     # Shall statistical dropout layers be used for tregularization purposes ? 
        , dropout_rate   = 0.25      # Rate for statistical dropout layer  
        , b_build_all    = False     # Added by RMO - full Model is build in 2 steps 
        ):
        
        '''
        Input: 
        The encoder_... and decoder_.... variables are Python lists,
        whose length defines the number of Conv2D and Conv2DTranspose layers 
        
        input_dim : Shape/dimensions of the input tensor - for MNIST (28,28,1) 
        encoder_conv_filters:     List with the number of maps/filters per Conv2D layer    
        encoder_conv_kernel_size: List with the kernel sizes for the Conv-Layers   
        encoder_conv_strides:     List with the strides used for the Conv-Layers   

        z_dim : dimension of the "latent_space"
        solution_type : Type of solution for KL loss calculation (0: Customized Encoder layer, 
                                                                  1: transfer of mu, var_log to Decoder 
                                                                  2: model.add_loss()
                                                                  3: definition of training step with Gradient.Tape()
        
        act :  determines activation function to use (0: LeakyRELU, 1:RELU , 2: SELU)
               !!!! NOTE: !!!!
               If SELU is used then the weight kernel initialization and the dropout layer need to be special   
               https://github.com/christianversloot/machine-learning-articles/blob/main/using-selu-with-tensorflow-and-keras.md
               AlphaDropout instead of Dropout + LeCunNormal for kernel initializer
        fact = 0.65e-4 : Factor to scale the KL loss relative to the reconstruction loss
                         Must be adapted to the way of calculation - 
                         e.g. for solution_type == 3 the loss is not averaged over all pixels 
                         => at least factor of around 1000 bigger than normally 
        loss-type = 0:   Defines the way we calculate a reconstruction loss 
                         0: Binary Cross Entropy - recommended by many authors 
                         1: Mean Square error - recommended by some authors especially for "face arithmetics"
        use_batch_norm = False   # True : We use BatchNormalization   
        use_dropout    = False   # True : We use dropout layers (rate = 0.25, see Encoder)
        b_build_all    = False   # True : Full VAE Model is build in 1 step; 
                                   False: Encoder, Decoder, VAE are build in separate steps   
        '''
        
        self.name = 'variational_autoencoder'

        # Parameters for Layers which define the Encoder and Decoder 
        self.input_dim                  = input_dim
        self.encoder_conv_filters       = encoder_conv_filters
        self.encoder_conv_kernel_size   = encoder_conv_kernel_size
        self.encoder_conv_strides       = encoder_conv_strides
        self.encoder_conv_padding       = encoder_conv_padding
        
        self.decoder_conv_t_filters     = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides     = decoder_conv_t_strides
        self.decoder_conv_t_padding     = decoder_conv_t_padding
        
        self.z_dim = z_dim

        # Check param for activation function 
        if act < 0 or act > 2: 
            print("Range error: Parameter act = " + str(act) + " has unknown value ")  
            sys.exit()
        else:
            self.act = act 
        
        # Factor to scale the KL loss relative to the Binary Cross Entropy loss 
        self.fact = fact 
        
        # Type of loss - 0: BCE, 1: MSE 
        self.loss_type = loss_type
        
        
        # Check param for solution approach  
        if solution_type < 0 or solution_type > 3: 
            print("Range error: Parameter solution_type = " + str(solution_type) + " has unknown value ")  
            sys.exit()
        else:
            self.solution_type = solution_type 

        self.use_batch_norm = use_batch_norm
        self.use_dropout    = use_dropout
        self.dropout_rate   = dropout_rate

        # Preparation of some variables to be filled later 
        self._encoder_input  = None  # receives the Keras object for the Input Layer of the Encoder 
        self._encoder_output = None  # receives the Keras object for the Output Layer of the Encoder 
        self.shape_before_flattening = None # info of the Encoder => is used by Decoder 
        self._decoder_input  = None  # receives the Keras object for the Input Layer of the Decoder
        self._decoder_output = None  # receives the Keras object for the Output Layer of the Decoder

        # Layers / tensors for KL loss 
        self.mu      = None # receives special Dense Layer's tensor for KL-loss 
        self.log_var = None # receives special Dense Layer's tensor for KL-loss 

        # Parameters for SELU - just in case we may need to use it somewhere 
        # https://keras.io/api/layers/activations/ see selu
        self.selu_scale = 1.05070098
        self.selu_alpha = 1.67326324

        # The number of Conv2D and Conv2DTranspose layers for the Encoder / Decoder 
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self.num_epoch = 0 # Intialization of the number of epochs 

        # A matrix for the values of the losses 
        self.std_loss  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # We only build the whole AE-model if requested
        self.b_build_all = b_build_all
        if b_build_all:
            self._build_all()


    def _build_enc(self, solution_type = -1, fact=-1.0):
        '''  Your documentation '''
        # Checking whether "fact" and "solution_type" for the KL loss shall be overwritten
        if fact < 0:
            fact = self.fact  
        if solution_type < 0:
            solution_type = self.solution_type
        else: 
            self.solution_type = solution_type
        
        # Preparation: We later need a function to calculate the z-points in the latent space 
        # The following function wiChangedll be used by an eventual Lambda layer of the Encoder 
        '''def z_point_sampling(args):
            
            A point in the latent space is calculated statistically 
            around an optimized mu for each sample 
            
            mu, log_var = args # Note: These are 1D tensors !
            epsilon = B.random_normal(shape=B.shape(mu), mean=0., stddev=1.)
            return mu + B.exp(log_var / 2) * epsilon'''

        
        # Input "layer"
        self._encoder_input = Input(shape=self.input_dim, name='encoder_input')

        # Initialization of a running variable x for individual layers 
        x = self._encoder_input
        # Build the CNN-part with Conv2D layers 
        # Note that stride>=2 reduces spatial resolution without the help of pooling layers 
        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'  # Important ! Controls the shape of the layer tensors.    
                , name = 'encoder_conv_' + str(i)
                )
            x = conv_layer(x)
            
            # The "normalization" should be done ahead of the "activation" 
            if self.use_batch_norm:
                x = BatchNormalization()(x)

            # Selection of activation function (out of 3)      
            if self.act == 0:
                x = LeakyReLU()(x)
            elif self.act == 1:
                x = ReLU()(x)
            elif self.act == 2: 
                # RMO: Just use the Activation layer to use SELU with predefined (!) parameters 
                x = Activation('selu')(x) 

            # Fulfill some SELU requirements 
            if self.use_dropout:
                if self.act == 2: 
                    x = AlphaDropout(rate = 0.25)(x)
                else:
                    x = Dropout(rate = 0.25)(x)
        x = identity_block(x,filter=self.encoder_conv_filters[self.n_layers_encoder-1])
        # Last multi-dim tensor shape - is later needed by the decoder 
        self._shape_before_flattening = B.int_shape(x)[1:]

        # Flattened layer before calculating VAE-output (z-points) via 2 special layers 
        x = Flatten()(x)
        
        # "Variational" part - create 2 Dense layers for a statistical distribution of z-points  
        self.mu      = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        if solution_type == 0: 
            # Customized layer for the calculation of the KL loss based on mu, var_log data
            # We use a customized layer according to a class definition  
            self.mu, self.log_var = My_KL_Layer()([self.mu, self.log_var], fact=fact)


        # Layer to provide a z_point in the Latent Space for each sample of the batch 
        #self._encoder_output = Lambda(z_point_sampling, name='encoder_output')([self.mu, self.log_var])
        self._encoder_output = CustomLambda()([self.mu, self.log_var])
        # The Encoder Model 
        # ~~~~~~~~~~~~~~~~~~~
        # With extra KL layer or with vae.add_loss()
        if self.solution_type == 0 or self.solution_type == 2: 
            self.encoder = Model(self._encoder_input, self._encoder_output, name="encoder")
        
        # Transfer solution => Multiple outputs 
        if self.solution_type == 1  or self.solution_type == 3: 
            self.encoder = Model(inputs=self._encoder_input, outputs=[self._encoder_output, self.mu, self.log_var], name="encoder")
            
    def _build_dec(self):
        ''' Your documentation       '''       
 
        # Input layer - aligned to the shape of z-points in the latent space = output[0] of the Encoder 
        self._decoder_inp_z = Input(shape=(self.z_dim,), name='decoder_input')
        
        # Additional Input layers for the KL tensors (mu, log_var) from the Encoder
        if self.solution_type == 1  or self.solution_type == 3: 
            self._dec_inp_mu       = Input(shape=(self.z_dim), name='mu_input')
            self._dec_inp_var_log  = Input(shape=(self.z_dim), name='logvar_input')
            
            # We give the layers later used as output a name 
            # Each of the Activation layers below just correspond to an identity passed through 
            #self._dec_mu            = self._dec_inp_mu 
            #self._dec_var_log       = self._dec_inp_var_log 
            self._dec_mu            = Activation('linear',name='dc_mu')(self._dec_inp_mu) 
            self._dec_var_log       = Activation('linear', name='dc_var')(self._dec_inp_var_log) 

        # Here we use the tensor shape info from the Encoder          
        x = Dense(np.prod(self._shape_before_flattening))(self._decoder_inp_z)
        x = Reshape(self._shape_before_flattening)(x)
        x = identity_block_transpose(x,filter=self.decoder_conv_t_filters[0])
        # The inverse CNN
        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same' # Important ! Controls the shape of tensors during reconstruction
                                   # we want an image with the same resolution as the original input 
                , name = 'decoder_conv_t_' + str(i)
                )
            x = conv_t_layer(x)

            # Normalization and Activation 
            if i < self.n_layers_decoder - 1:
                # Also in the decoder: normalization before activation  
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                
                # Choice of activation function
                if self.act == 0:
                    x = LeakyReLU()(x)
                elif self.act == 1:
                    x = ReLU()(x)
                elif self.act == 2: 
                    #x = self.selu_scale * ELU(alpha=self.selu_alpha)(x)
                    x = Activation('selu')(x)
                
                # Adaptions to SELU requirements 
                if self.use_dropout:
                    if self.act == 2: 
                        x = AlphaDropout(rate = 0.25)(x)
                    else:
                        x = Dropout(rate = 0.25)(x)
                
            # Last layer => Sigmoid output 
            # => This requires s<pre style="padding:8px; height: 400px; overflow:auto;">caled input => Division of pixel values by 255
            else:
                x = Activation('sigmoid', name='dc_reco')(x)

        # Output tensor => a scaled image 
        self._decoder_output = x

        # The Decoder model 
        # solution_type == 0/2/3: Just the decoded input 
        if self.solution_type == 0 or self.solution_type == 2 or self.solution_type == 3: 
            self.decoder = Model(self._decoder_inp_z, self._decoder_output, name="decoder")
        
        # solution_type == 1: The decoded tensor plus the transferred tensors mu and log_var a for the variational distribution 
        if self.solution_type == 1: 
            self.decoder = Model([self._decoder_inp_z, self._dec_inp_mu, self._dec_inp_var_log], 
                                 [self._decoder_output, self._dec_mu, self._dec_var_log], name="decoder")
    # Function to build the full VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    def _build_VAE(self):     
        ''' Your documentation '''
        
        # Solution with train_step() and GradientTape(): Control is transferred to class VAE  
        if self.solution_type == 3:
            self.model = VAE(self)   # Here parameter "self" provides a reference to an instance of MyVariationalAutoencoder
            #self.model.summary()
        
        # Solutions with layer.add_loss or model.add_loss() 
        if self.solution_type == 0 or self.solution_type == 2:
            model_input  = self._encoder_input
            model_output = self.decoder(self._encoder_output)
            self.model = Model(model_input, model_output, name="vae")

        # Solution with transfer of data from the Encoder to the Decoder output layer
        if self.solution_type == 1: 
            enc_out      = self.encoder(self._encoder_input)
            dc_reco, dc_mu, dc_var = self.decoder(enc_out)
            # We organize the output and later association of cost functions and metrics via a dictionary 
            mod_outputs = {'vae_out_main': dc_reco, 'vae_out_mu': dc_mu, 'vae_out_var': dc_var}
            self.model = Model(inputs=self._encoder_input, outputs=mod_outputs, name="vae")

    # Function to build full AE in one step if requested
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _build_all(self):
        self._build_enc()
        self._build_dec()
        self._build_VAE()
    # Function to compile VA-model with a KL-layer in the Encoder 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compile_for_KL_Layer(self, learning_rate):
        if self.solution_type != 0: 
            print("The compile_L() function is only compatible with solution_type = 0")
            sys.exit()
        self.learning_rate = learning_rate
        # Optimizer 
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy",
                           metrics=[tf.keras.metrics.BinaryCrossentropy(name='bce')])
    def train_model_with_KL_Layer(self, x_train, batch_size, epochs, initial_epoch = 0):
        self.model.fit(     
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
        )

    # Function to compile the full VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    def compile_myVAE(self, learning_rate):
    
        # Optimizer
        # ~~~~~~~~~ 
        optimizer = Adam(learning_rate=learning_rate)
        # save the learning rate for possible intermediate output to files 
        self.learning_rate = learning_rate
        
        # Parameter "fact" will be used by the cost functions defined below to scale the KL loss relative to the BCE loss 
        fact = self.fact
        
        # Function for solution_type == 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        @tf.function
        def mu_loss(y_true, y_pred):
            loss_mux = fact * tf.reduce_mean(tf.square(y_pred))
            return loss_mux
        
        @tf.function
        def logvar_loss(y_true, y_pred):
            loss_varx = -fact * tf.reduce_mean(1 + y_pred - tf.exp(y_pred))    
            return loss_varx

        # Function for solution_type == 2 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We follow an approach described at  
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
        # NOTE: We can NOT use @tf.function here 
        def get_kl_loss(mu, log_var):
            kl_loss = -fact * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
            return kl_loss


        # Required operations for solution_type==2 => model.add_loss()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        res_kl = get_kl_loss(mu=self.mu, log_var=self.log_var)

        if self.solution_type == 2: 
            self.model.add_loss(res_kl)    
            self.model.add_metric(res_kl, name='kl', aggregation='mean')
        
        # Model compilation 
        # ~~~~~~~~~~~~~~~~~~~~
        
        # Solutions with layer.add_loss or model.add_loss() 
        if self.solution_type == 0 or self.solution_type == 2: 
            if self.loss_type == 0: 
                self.model.compile(optimizer=optimizer, loss="binary_crossentropy",
                                   metrics=[tf.keras.metrics.BinaryCrossentropy(name='bce')])
            if self.loss_type == 1: 
                self.model.compile(optimizer=optimizer, loss="mse",
                                   metrics=[tf.keras.metrics.MSE(name='mse')])
        
        # Solution with transfer of data from the Encoder to the Decoder output layer
        if self.solution_type == 1: 
            if self.loss_type == 0: 
                self.model.compile(optimizer=optimizer
                                   , loss={'vae_out_main':'binary_crossentropy', 'vae_out_mu':mu_loss, 'vae_out_var':logvar_loss} 
                                   #, metrics={'vae_out_main':tf.keras.metrics.BinaryCrossentropy(name='bce'), 'vae_out_mu':mu_loss, 'vae_out_var': logvar_loss }
                                   )
            if self.loss_type == 1: 
                self.model.compile(optimizer=optimizer
                                   , loss={'vae_out_main':'mse', 'vae_out_mu':mu_loss, 'vae_out_var':logvar_loss} 
                                   #, metrics={'vae_out_main':tf.keras.metrics.MSE(name='mse'), 'vae_out_mu':mu_loss, 'vae_out_var': logvar_loss }
                                   )
       
        # Solution with train_step() and GradientTape(): Control is transferred to class VAE  
        if self.solution_type == 3:
            self.model.compile(optimizer=optimizer)
    # Function to initiate training 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train_myVAE(self, x_train, x_target=None
                    , b_use_generator   = False 
                    , b_target_ne_train = False
                    , batch_size = 32
                    , epochs = 2
                    , initial_epoch = 0, 
                    t_mu=None, 
                    t_logvar=None
                    ,callbacks = None
                    ):

        ''' 
        @note: Sometimes x_target MUST be provided - e.g. for Denoising, Recolorization 
        @note: x_train will come as a dataflow in case of a generator 
        '''

        # cax = ProgbarLogger(count_mode='samples', stateful_metrics=None)
        
        class MyPrinterCallback(tf.keras.callbacks.Callback):
            # def on_train_batch_begin(self, batch, logs=None):
            #     # Do something on begin of training batch
        
            def on_epoch_end(self, epoch, logs=None):
                # Get overview over available keys 
                #keys = list(logs.keys())
                print("\nEPOCH: {}, Total Loss: {:8.6f}, // reco loss: {:8.6f}, mu Loss: {:8.6f}, logvar loss: {:8.6f}".format(epoch, 
                      logs['loss'], logs['decoder_loss'], logs['decoder_1_loss'], logs['decoder_2_loss'] 
                                            ))
                print()
                #print('EPOCH: {}, Total Loss: {}'.format(epoch, logs['loss']))
                #print('EPOCH: {}, metrics: {}'.format(epoch, logs['metrics']))
        
            def on_epoch_begin(self, epoch, logs=None):
                print('-'*50)
                print('STARTING EPOCH: {}'.format(epoch))
                
        if not b_target_ne_train : 
            x_target = x_train

        # Data are provided from tensors in the Video RAM 
        if not b_use_generator: 

            # Solutions with layer.add_loss or model.add_loss() 
            # Solution with train_step() and GradientTape(): Control is transferred to class VAE  
            if self.solution_type == 0 or self.solution_type == 2 or self.solution_type == 3: 
                self.model.fit(     
                    x_train
                    , x_target
                    , batch_size = batch_size
                    , shuffle = True
                    , epochs = epochs
                    , initial_epoch = initial_epoch
                )
            
            # Solution with transfer of data from the Encoder to the Decoder output layer
            if self.solution_type == 1: 
                self.model.fit(     
                    x_train
                    , {'vae_out_main': x_target, 'vae_out_mu': t_mu, 'vae_out_var':t_logvar}
                    #               also working  
                    #                , [x_train, t_mu, t_logvar] # we provide some dummy tensors here  
                    , batch_size = batch_size
                    , shuffle = True
                    , epochs = epochs
                    , initial_epoch = initial_epoch
                    #, verbose=1
                )
    
        # If data are provided as a batched dataflow from a generator - e.g. for Celeb A 
        else: 

            # Solution with transfer of data from the Encoder to the Decoder output layer
            if self.solution_type == 1: 
                print("We have no solution yet for solution_type==1 and generators !")
                sys.exit()

            # Solutions with layer.add_loss or model.add_loss() 
            # Solution with train_step() and GradientTape(): Control is transferred to class VAE  
            if self.solution_type == 0 or self.solution_type == 2 or self.solution_type == 3: 
                self.model.fit(     
                    x_train   # coming as a batched dataflow from the outside generator - no batch size required here 
                    , shuffle = True
                    , epochs = epochs
                    , initial_epoch = initial_epoch
                    , callbacks = callbacks
                )
                
                
# RUN TO LOAD NPY DATASET TO TRAIN AND VAL
for game in finalgames:
    screensize = 160
    PROJECT_ROOT_DIR = "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/DATASETS/outputs/kerasimgs/"
    GAME = game
    print("Training the VAE on", GAME)
    TRAIN_NAME = GAME+"_train"
    VAL_NAME = GAME+"_val"
    SIZE = str(screensize)+'_'+str(screensize)
    EXT = ".npy" 
    TRAIN_PATH = PROJECT_ROOT_DIR+GAME+'/'+SIZE+'/'+TRAIN_NAME+EXT
    VAL_PATH = PROJECT_ROOT_DIR+GAME+'/'+SIZE+'/'+VAL_NAME+EXT
    train = np.load(TRAIN_PATH)
    val = np.load(VAL_PATH)
    print("train", train.shape)
    print("val", val.shape)
    b_use_generator_ay = True
    BATCH_SIZE    = 256
    SOLUTION_TYPE = 3

    if b_use_generator_ay:
        # solution_type == 0 works with extra layers and add_loss to control the KL loss
        # it requires the definition of "labels" - which are the original images  
        if SOLUTION_TYPE == 0: 
            data_gen = ImageDataGenerator()
            data_flow = data_gen.flow(
                            train 
                            , train
                            #, target_size = INPUT_DIM[:2]
                            , batch_size = BATCH_SIZE
                            , shuffle = True
                            #, class_mode = 'input'   # Not working with this type of generator 
                            #, subset = "training"    # Not required 
                            )

        if SOLUTION_TYPE == 3: 
            data_gen = ImageDataGenerator()
            data_flow = data_gen.flow(
                            train 
                            , batch_size = BATCH_SIZE
                            , shuffle = True
                            )
            
    MyVae = MyVariationalAutoencoder(
    input_dim = INPUT_DIM
   , encoder_conv_filters     = [32,32,64,]
   , encoder_conv_kernel_size = [1,4,4]
   , encoder_conv_strides     = [1,2,2]
   , encoder_conv_padding     = ['same','same','same']


   , decoder_conv_t_filters     = [64,64,n_ch]
   , decoder_conv_t_kernel_size = [1,3,3]
   , decoder_conv_t_strides     = [1,2,2]
   , decoder_conv_t_padding     = ['same','same','same']

    , z_dim = z_dim
    , solution_type = solution_type    
    , act   = act
    , fact  = fact
    , loss_type      = loss_type
    , use_batch_norm = use_batch_norm
    , use_dropout    = use_dropout
    , dropout_rate   = dropout_rate
)
    
    MyVae._build_enc()
    MyVae._build_dec()
    MyVae._build_VAE()
    learning_rate = 0.0001
    MyVae.compile_myVAE(learning_rate=learning_rate)
    with tf.device('/GPU:0'):
        MyVae.train_myVAE(   
                    data_flow
                    , b_use_generator = True 
                    , epochs = n_epochs
                    , initial_epoch = INITIAL_EPOCH
                    , callbacks=[#WandbMetricsLogger(log_freq=1),
                                #WandbModelCheckpoint("models"),
                                keras.callbacks.EarlyStopping(monitor="total_loss",min_delta=1,patience=5,mode='min',restore_best_weights=True)]
                    )
    modelpath = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/newmodels/'
    #MyVae.encoder.save('/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/newmodels/resvaeplus/z'+str(z_dim)+'/'+GAME+'/encoder')
    #MyVae.decoder.save('/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/newmodels/resvaeplus/z'+str(z_dim)+'/'+GAME+'/decoder')
    MyVae.encoder.save(modelpath+GAME+'/encoder')
    MyVae.decoder.save(modelpath+GAME+'/decoder')
    print("VAE FOR",GAME,"SAVED")
    
print("ALL VAEs TRAINED AND SAVED")