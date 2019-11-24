import numpy as np
from matplotlib import pyplot as plt
from gym.core import Env
from gym.spaces import Box
from torchvision.datasets import MNIST

class DrawEnv(Env):
    def __init__(self):

        self.window_x = [-10.0,10.0]
        self.window_y = [-10.0,10.0]

        self.res = 28
        self.counter = 0
        self.canvas = np.zeros([self.res,self.res],dtype=np.float32) #- 1.0
        self.target = np.ones([self.res,self.res],dtype=np.float32)

        self.pad_width = 4  ## image padding so it works easily with stable_baselines

        self.action_space = Box(low=np.array([self.window_x[0],
                                              self.window_y[0],
                                              self.window_x[0],
                                              self.window_y[0],
                                              1.0,
                                              0.0]),
                                high=np.array([self.window_x[1],
                                               self.window_y[1],
                                               self.window_x[1],
                                               self.window_y[1],
                                               3.0,
                                               1.0]),
                                dtype=np.float32)

        ### FOR MLP
        #lows = np.zeros(784*2+1)
        #highs = np.concatenate([[np.inf],np.ones(784*2)])
        #self.observation_space = Box(low=lows,
        #                             high=highs,dtype=np.float32)

        ### FOR CNN
        self.observation_space = Box(0,1,[28+2*self.pad_width,
                                          (28+2*self.pad_width)*2,1])

        dataset = MNIST("./data/", download=True)
        self.images = dataset.train_data.numpy().reshape([60000, 28, 28, 1])/255.0

        self.setup_target()



    def setup_target(self):
        idx = np.random.randint(0,self.images.shape[0]-1)

        #self.target = self.images[0,:,:]  # MLP

        # CNN
        #tgt = np.pad(self.images[0,:,:,0],self.pad_width).reshape([28+2*self.pad_width,
        #                                                           28+2*self.pad_width,
        #                                                           1])
        #self.target = tgt
        self.target = self.images[idx,:,:,:] # CNN




    def step(self, a):
        self.counter = self.counter + 1

        ###### break down the observation
        start_point = a[0:2]
        stop_point = a[2:4]
        thickness = np.int(np.round(np.abs(a[4])))
        color = a[5]

        ##############

        paint_res = 50
        for i in range(paint_res):
            curpoint = [start_point[0] * (1-i/(paint_res-1)) +
                        stop_point[0] * (i/(paint_res-1)),
                        start_point[1] * (1-i/(paint_res-1)) +
                        stop_point[1] * (i/(paint_res-1))]
            matpoint = self._window_to_matrix(curpoint)


            self.canvas[matpoint[0]:(matpoint[0] + thickness),
                        matpoint[1]:(matpoint[1] + thickness)] = color

        #reward = -np.sum(np.abs(self.canvas - self.target)) ## MLP
        #reward = -np.sum(np.square(self.canvas - self.target[:,:,0]))
        reward1 = np.sum(self.canvas[self.target[:,:,0] > 0.1] > 0.1)
        reward0 = np.sum(self.canvas[self.target[:,:,0] <= 0.1] <= 0.1)
        reward = reward1 - 0.1 * reward0
        done = self.counter > 10
        ob = self._get_obs()


        return ob, reward, done, {}

    def _window_to_matrix(self,wpt):



        center_origin = [1.0/(self.res*2) * (self.window_x[1] - self.window_x[0]) + self.window_x[0],
                         1.0/(self.res*2) * (self.window_y[1] - self.window_y[0]) + self.window_y[0]]

        mat_x = np.int(round((wpt[0] - center_origin[0])/(self.window_x[1] - self.window_x[0]) * (self.res-1)))
        mat_y = np.int(round((wpt[1] - center_origin[1])/(self.window_y[1] - self.window_y[0]) * (self.res-1)))

        return [mat_x,mat_y]

    def _get_obs(self):
        output_mlp = np.concatenate([[self.counter],self.canvas.flatten(),self.target.flatten()])

        output_cnn = np.concatenate([
            np.pad(
                np.pad(self.canvas,self.pad_width).reshape([28+2*self.pad_width,28+2*self.pad_width,1]),
                [(0, 0), (0, 0), (0, 0)]),
            np.pad(
            np.pad(self.target[:,:,0],self.pad_width).reshape([28+2*self.pad_width,28+2*self.pad_width,1]),
                [(0, 0), (0, 0), (0, 0)])
        ],axis=1)

        return output_cnn

    def get_formatted_obs(self):
        return (self.counter, self.canvas, self.target)

    def render(self, mode='human'):
        #plt.subplot(2,1,1)
        #plt.imshow(self.canvas)
        #plt.subplot(2, 1, 2)
        #plt.imshow(self.target[:,:,0])
        plt.imshow(self._get_obs()[:,:,0])

    def reset(self):
        self.counter = 0
        self.canvas = np.zeros([28, 28])
        self.target = np.ones([28, 28, 1])

        self.setup_target()

        return self._get_obs()
