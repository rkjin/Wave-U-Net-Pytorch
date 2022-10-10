import torch
import torch.nn as nn

from model.crop import centre_crop
from model.resample import Resample1d
from model.conv import ConvLayer

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING  
        if res == "fixed": #True
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed": #True
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)
            # print('conv0',conv)
        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)
            # print('conv1',conv)
        # DOWNSAMPLING
        out = self.downconv(out)
        # print('downconv', self.downconv)
        return out, shortcut
        # conv0 ConvLayer(
        # (filter): Conv1d(2, 32, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(4, 32, eps=1e-05, affine=True)
        # )
        # conv1 ConvLayer(
        # (filter): Conv1d(32, 64, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
        # )
        # downconv Resample1d()
        # conv0 ConvLayer(
        # (filter): Conv1d(64, 64, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
        # )
        # conv1 ConvLayer(
        # (filter): Conv1d(64, 128, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
        # )
        # downconv Resample1d()
        # conv0 ConvLayer(
        # (filter): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
        # )
        # conv1 ConvLayer(
        # (filter): Conv1d(128, 256, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )
        # downconv Resample1d()
        # conv0 ConvLayer(
        # (filter): Conv1d(256, 256, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )
        # conv1 ConvLayer(
        # (filter): Conv1d(256, 512, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
        # )
        # downconv Resample1d()
        # conv0 ConvLayer(
        # (filter): Conv1d(512, 512, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
        # )
        # conv1 ConvLayer(
        # (filter): Conv1d(512, 1024, kernel_size=(5,), stride=(1,))
        # (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
        # )
        # downconv Resample1d()
    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, conv_type, res, separate=False, depth=1, strides=2):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels) #6 #[32, 64, 128, 256, 512, 1024]
        self.strides = strides #4
        self.kernel_size = kernel_size #5
        self.num_inputs = num_inputs # 2 ch
        self.num_outputs = num_outputs #2 ch
        self.depth = depth #1
        self.instruments = instruments #  ['bass', 'drums', 'other', 'vocals']
        self.separate = separate # 1
        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()
        model_list = instruments if separate else ["ALL"] #  ['bass', 'drums', 'other', 'vocals']
 
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list: #  ['bass', 'drums', 'other', 'vocals']
            module = nn.Module() #??????????????????????????????????????????
            module.downsampling_blocks = nn.ModuleList() #??????????????????????????????????????????
            module.upsampling_blocks = nn.ModuleList() #??????????????????????????????????????????

            for i in range(self.num_levels - 1): # 6-1 = 5
                in_ch = num_inputs if i == 0 else num_channels[i]
#               print(i, in_ch) 0:2, 1:64, 2:128, 3:246, 4:512
                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)                                                    
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module
 
        self.set_output_size(target_output_size) #88200 = 44.1khz * 2 

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)      
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
             # Using valid convolutions with 97961 inputs and 88409 outputs
        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}
     # {'output_start_frame': 4776, 'output_end_frame': 93185, 'output_frames': 88409, 'input_frames': 97961}
    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x #([4, 2, 97961])

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short) #len = 5
            #print(out.shape, short.shape)
            # torch.Size([4, 64, 24489]) torch.Size([4, 32, 97957])
            # torch.Size([4, 128, 6121]) torch.Size([4, 64, 24485])
            # torch.Size([4, 256, 1529]) torch.Size([4, 128, 6117])
            # torch.Size([4, 512, 381]) torch.Size([4, 256, 1525])
            # torch.Size([4, 1024, 94]) torch.Size([4, 512, 377])
            # block DownsamplingBlock(
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(2, 32, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(4, 32, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(32, 64, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
            #     )
            # )
            # (downconv): Resample1d()
            # )
            # block DownsamplingBlock(
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(64, 64, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(64, 128, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
            #     )
            # )
            # (downconv): Resample1d()
            # )
            # block DownsamplingBlock(
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(128, 256, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
            #     )
            # )
            # (downconv): Resample1d()
            # )
            # block DownsamplingBlock(
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(256, 256, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(256, 512, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
            #     )
            # )
            # (downconv): Resample1d()
            # )
            # block DownsamplingBlock(
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(512, 512, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(512, 1024, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
            #     )
            # )
            # (downconv): Resample1d()
            # )
        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)
            # print(out.shape)
            # torch.Size([4, 1024, 90])
            # torch.Size([4, 1024, 90])
            # torch.Size([4, 1024, 90])
            # torch.Size([4, 1024, 90])
            # print('conv',conv)
            # conv ConvLayer(
            # (filter): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,))
            # (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
            # )
            # conv ConvLayer(
            # (filter): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,))
            # (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
            # )
            # conv ConvLayer(
            # (filter): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,))
            # (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
            # )
            # conv ConvLayer(
            # (filter): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,))
            # (norm): GroupNorm(128, 1024, eps=1e-05, affine=True)
            # )
        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            print('up block', block)
            # print(idx, out.shape, shortcuts[-1-idx].shape)
            out = block(out, shortcuts[-1 - idx])
            # print(idx, out.shape)
            # 0 torch.Size([4, 1024, 90]) torch.Size([4, 512, 377])
            # 0 torch.Size([4, 512, 349])
            # 1 torch.Size([4, 512, 349]) torch.Size([4, 256, 1525])
            # 1 torch.Size([4, 256, 1385])
            # 2 torch.Size([4, 256, 1385]) torch.Size([4, 128, 6117])
            # 2 torch.Size([4, 128, 5529])
            # 3 torch.Size([4, 128, 5529]) torch.Size([4, 64, 24485])
            # 3 torch.Size([4, 64, 22105])
            # 4 torch.Size([4, 64, 22105]) torch.Size([4, 32, 97957])
            # 4 torch.Size([4, 32, 88409])
            # print('idx, block',idx, block)
            # idx, block 0 UpsamplingBlock(
            # (upconv): Resample1d()
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(1024, 512, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(1024, 512, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(64, 512, eps=1e-05, affine=True)
            #     )
            # )
            # )
            # idx, block 1 UpsamplingBlock(
            # (upconv): Resample1d()
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(512, 256, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(512, 256, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(32, 256, eps=1e-05, affine=True)
            #     )
            # )
            # )
            # idx, block 2 UpsamplingBlock(
            # (upconv): Resample1d()
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(256, 128, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(256, 128, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(16, 128, eps=1e-05, affine=True)
            #     )
            # )
            # )
            # idx, block 3 UpsamplingBlock(
            # (upconv): Resample1d()
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(128, 64, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(128, 64, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(8, 64, eps=1e-05, affine=True)
            #     )
            # )
            # )
            # idx, block 4 UpsamplingBlock(
            # (upconv): Resample1d()
            # (pre_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(64, 32, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(4, 32, eps=1e-05, affine=True)
            #     )
            # )
            # (post_shortcut_convs): ModuleList(
            #     (0): ConvLayer(
            #     (filter): Conv1d(64, 32, kernel_size=(5,), stride=(1,))
            #     (norm): GroupNorm(4, 32, eps=1e-05, affine=True)
            #     )
            # )
            # )            
        # OUTPUT CONV
        # print(out.shape)
        out = module.output_conv(out)  
        # print(out.shape)
        # torch.Size([4, 32, 88409])
        # torch.Size([4, 2, 88409])
        # print('module', module.output_conv)
        # module Conv1d(32, 2, kernel_size=(1,), stride=(1,))

        if not self.training:  # False      At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out #([4, 2, 88409])

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1] # [4, 2, 97961]
        assert(curr_input_size == self.input_size) # 97961 inputs and 88409 outputs
        if self.separate: #True
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict