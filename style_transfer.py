""" Implementation in TensorFlow of the paper
A Neural Algorithm of Artistic Style (Gatys et al., 2016)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

For more details, please read the assignment handout:
https://docs.google.com/document/d/1FpueD-3mScnD0SJQDtwmOb1FrSwo1NGowkXzMwPoLH4/edit?usp=sharing
"""

local_flag=0
skywalker = 1 - local_flag

import os

if local_flag:
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    input_path = ''
    checkpoint_path = 'checkpoints'
    outputs_path = 'outputs'

if skywalker:
    input_path = '/input/yoav_ST/'
    checkpoint_path = os.environ['OUTPUT'] + '/checkpoints'
    outputs_path = os.environ['OUTPUT'] + '/outputs'

import time

import numpy as np
import tensorflow as tf

import load_vgg
import utils


def setup():

    utils.safe_mkdir(checkpoint_path)
    utils.safe_mkdir(outputs_path)



class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
        # content_img and style_img should be strings with the path to the image
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

        ###############################
        ## TO DO
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        # content_w, style_w: corresponding weights for content loss and style loss
        self.content_w = 0.05
        self.style_w = 1
        # style_layer_w: weights for different style layers. deep layers have more weights
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]
        self.gstep = tf.Variable(0, trainable=False, name='global_step')  # global step
        self.lr = 2
        ###############################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image,
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time,
        training the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('in_img',
                                             shape=([1, self.img_height, self.img_width, 3]),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''
        self.vgg = load_vgg.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        ''' Calculate the loss between the feature representation of the
        content image and the generated image.

        Inputs:
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        '''
        ###############################
        ## TO DO
        #layer_shape = tf.shape(P)
        #coeff = 4*(tf.reduce_prod(layer_shape[1:]))
        #squared_norm = tf.square(tf.norm(F-P))
        #self.content_loss = tf.divide(squared_norm, tf.cast(coeff, dtype=tf.float32))
        self.content_loss = tf.reduce_sum((F-P)**2) / (4.0*P.size)
        ###############################

    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
            Hint: you'll first have to reshape F
        """
        ###############################
        ## TO DO
        # N - number of filters (third dimension of F
        # M - number of features (multiplication of first two dimensions of F)
        # gram matrix is of dimension NxN

        reshaped = tf.reshape(F,[M,N])
        return tf.matmul(reshaped,reshaped,transpose_a=True)
        ###############################

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)

        Hint: 1. you'll have to use the function _gram_matrix()
            2. we'll use the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        ###############################
        ## TO DO
        with tf.variable_scope('ssl',reuse=tf.AUTO_REUSE) as scope:
            #M = tf.shape(a)[3]
            #N = tf.reduce_prod(tf.shape(a)[1:3])

            N = a.shape[3]
            M = a.shape[1] * a.shape[2]

            gram_g = self._gram_matrix(g, N, M)
            gram_a = self._gram_matrix(a, N, M)

            #squared_norm = tf.square(tf.norm(gram_g - gram_a))

            #M = tf.cast(M,tf.float64)
            #N = tf.cast(N, tf.float64)
            #coeff = 4*tf.square(M)*tf.square(N)

            #return tf.divide(squared_norm, tf.cast(coeff, tf.float32))

            return tf.reduce_sum((gram_g - gram_a)**2) / (4.0*((M*N)**2))
        ###############################

    def _style_loss(self, A):
        """ Calculate the total style loss as a weighted sum
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        ###############################
        ## TO DO
        G = [getattr(self.vgg, self.style_layers[i]) for i in range(5)]


        self.style_loss = sum([self.style_layer_w[i]*self._single_style_loss(A[i], G[i]) for i in range(5)])
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                # assign content image to the input variable
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])
            self._style_loss(style_layers)

            ##########################################
            ## TO DO: create total loss.
            ## Hint: don't forget the weights for the content loss and style loss
            self.total_loss = self.content_w * self.content_loss +  self.style_w * self.style_loss
            ##########################################


    def optimize(self):
        ###############################
        ## TO DO: create optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt = self.optimizer.minimize(self.total_loss, global_step=self.gstep)


        ###############################

    def create_summary(self):
        ###############################
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        self.summary_op = None
        ###############################

    def build(self):
        print("creating input...")
        self.create_input()
        print("loading vgg...")
        self.load_vgg()
        print("establishing losses...")
        self.losses()
        print("establishing optimizer...")
        self.optimize()
        print("creating summaries")
        self.create_summary()

    def train(self, n_iters):
        print ("training...")
        skip_step = 1
        with tf.Session() as sess:

            ###############################
            ## TO DO:
            ## 1. initialize your variables
            ## 2. create writer to write your grapp
            init = tf.global_variables_initializer()
            sess.run(init)

            ###############################

            sess.run(self.input_img.assign(self.initial_img))

            ###############################
            ## TO DO:
            ## 1. create a saver object
            ## 2. check if a checkpoint exists, restore the variables

            saver = tf.train.Saver()

            checkpoint_exists = tf.train.get_checkpoint_state(checkpoint_path)
            if checkpoint_exists:
                saver.restore(sess, checkpoint_exists.model_checkpoint_path)

            ##############################

            initial_step = self.gstep.eval()

            #utils.save_image('outputs/content.png', self.input_img.assign(self.initial_img).eval()+self.vgg.mean_pixels)
            start_time = time.time()
            for index in range(initial_step, n_iters):



                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20

                print("running opt")
                sess.run(self.opt)

                # Initial 5 steps are collected and the progress is printed, then step 10 and then 20, 40, etc.
                if (index + 1) % skip_step == 0:
                    print("step", index + 1, "complete")
                    ###############################
                    ## TO DO: obtain generated image, loss, and summary

                    gen_image, total_loss, summary =  self.input_img.eval(), self.total_loss.eval(), None

                    ###############################

                    # add back the mean pixels we subtracted before
                    gen_image = gen_image + self.vgg.mean_pixels
                    #writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    print('   Loss: {:5.1f}'.format(total_loss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()



                    if (index + 1) % 20 == 0 or (index+1) % n_iters == 0:
                        ###############################
                        ## TO DO: save the variables into a checkpoint
                        print("Saving...")

                        filename = outputs_path + '/%d.png' % (index)
                        utils.save_image(filename, gen_image)

                        saver.save(sess=sess, save_path= checkpoint_path + '/style_transfer',global_step=self.gstep)

                        ###############################
                        pass


if __name__ == '__main__':
    setup()
    machine = StyleTransfer(input_path + 'content/punch.jpg', input_path + 'styles/picasso.jpeg', 666,500)#333, 250)
    machine.build()
    machine.train(1000)