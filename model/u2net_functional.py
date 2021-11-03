import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D


def ConvBlock(inputs, out_ch=3, dirate=1):
    conv = Conv2D(out_ch, (3, 3), strides=1, padding='same', dilation_rate=dirate)
    bn = BatchNormalization()
    relu = ReLU()
    return relu(bn(conv(inputs)))


def RSU7(inputs, mid_ch=12, out_ch=3):
    conv_b0 = ConvBlock(out_ch, dirate=1)

    conv_b1 = ConvBlock(mid_ch, dirate=1)
    pool1   = MaxPool2D(2, strides=(2, 2))

    conv_b2 = ConvBlock(mid_ch, dirate=1)
    pool2   = MaxPool2D(2, strides=(2, 2))

    conv_b3 = ConvBlock(mid_ch, dirate=1)
    pool3   = MaxPool2D(2, strides=(2, 2))

    conv_b4 = ConvBlock(mid_ch, dirate=1)
    pool4   = MaxPool2D(2, strides=(2, 2))

    conv_b5 = ConvBlock(mid_ch, dirate=1)
    pool5   = MaxPool2D(2, strides=(2, 2))

    conv_b6 = ConvBlock(mid_ch, dirate=1)
    conv_b7 = ConvBlock(mid_ch, dirate=2)

    conv_b6_d  = ConvBlock(mid_ch, dirate=1)
    upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b5_d  = ConvBlock(mid_ch, dirate=1)
    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b4_d  = ConvBlock(mid_ch, dirate=1)
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b3_d  = ConvBlock(mid_ch, dirate=1)
    upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b2_d  = ConvBlock(mid_ch, dirate=1)
    upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b1_d  = ConvBlock(out_ch, dirate=1)
    upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    hx = inputs
    hxin = conv_b0(hx)

    hx1 = conv_b1(hxin)
    hx = pool1(hx1)

    hx2 = conv_b2(hx)
    hx = pool2(hx2)

    hx3 = conv_b3(hx)
    hx = pool3(hx3)

    hx4 = conv_b4(hx)
    hx = pool4(hx4)

    hx5 = conv_b5(hx)
    hx = pool5(hx5)

    hx6 = conv_b6(hx)

    hx7 = conv_b7(hx6)

    hx6d = conv_b6_d(tf.concat([hx7, hx6], axis=3))
    hx6dup = upsample_5(hx6d)

    hx5d = conv_b5_d(tf.concat([hx6dup, hx5], axis=3))
    hx5dup = upsample_4(hx5d)

    hx4d = conv_b4_d(tf.concat([hx5dup, hx4], axis=3))
    hx4dup = upsample_3(hx4d)

    hx3d = conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
    hx3dup = upsample_2(hx3d)

    hx2d =  conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
    hx2dup = upsample_1(hx2d)

    hx1d = conv_b1_d(tf.concat([hx2dup, hx1], axis=3))
    
    return hx1d + hxin


def RSU6(inputs, mid_ch=12, out_ch=3):
    conv_b0 = ConvBlock(out_ch, dirate=1)

    conv_b1 = ConvBlock(mid_ch, dirate=1)
    pool1   = MaxPool2D(2, strides=(2, 2))

    conv_b2 = ConvBlock(mid_ch, dirate=1)
    pool2   = MaxPool2D(2, strides=(2, 2))

    conv_b3 = ConvBlock(mid_ch, dirate=1)
    pool3   = MaxPool2D(2, strides=(2, 2))

    conv_b4 = ConvBlock(mid_ch, dirate=1)
    pool4   = MaxPool2D(2, strides=(2, 2))

    conv_b5 = ConvBlock(mid_ch, dirate=1)
    pool5   = MaxPool2D(2, strides=(2, 2))

    conv_b6 = ConvBlock(mid_ch, dirate=2)

    conv_b5_d = ConvBlock(mid_ch, dirate=1)
    upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b4_d = ConvBlock(mid_ch, dirate=1)
    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b3_d = ConvBlock(mid_ch, dirate=1)
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b2_d = ConvBlock(mid_ch, dirate=1)
    upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b1_d = ConvBlock(out_ch, dirate=1)
    upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    hx = inputs
    hxin = conv_b0(hx)

    hx1 = conv_b1(hxin)
    hx = pool1(hx1)

    hx2 = conv_b2(hx)
    hx = pool2(hx2)

    hx3 = conv_b3(hx)
    hx = pool3(hx3)

    hx4 = conv_b4(hx)
    hx = pool4(hx4)

    hx5 = conv_b5(hx)

    hx6 = conv_b6(hx5)

    hx5d = conv_b5_d(tf.concat([hx6, hx5], axis=3))
    hx5dup = upsample_4(hx5d)

    hx4d = conv_b4_d(tf.concat([hx5dup, hx4], axis=3))
    hx4dup = upsample_3(hx4d)

    hx3d = conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
    hx3dup = upsample_2(hx3d)

    hx2d =  conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
    hx2dup = upsample_1(hx2d)

    hx1d = conv_b1_d(tf.concat([hx2dup, hx1], axis=3))
    
    return hx1d + hxin


def RSU5(inputs, mid_ch=12, out_ch=3):
    conv_b0 = ConvBlock(out_ch, dirate=1)

    conv_b1 = ConvBlock(mid_ch, dirate=1)
    pool1   = MaxPool2D(2, strides=(2, 2))

    conv_b2 = ConvBlock(mid_ch, dirate=1)
    pool2   = MaxPool2D(2, strides=(2, 2))

    conv_b3 = ConvBlock(mid_ch, dirate=1)
    pool3   = MaxPool2D(2, strides=(2, 2))

    conv_b4 = ConvBlock(mid_ch, dirate=1)
    pool4   = MaxPool2D(2, strides=(2, 2))

    conv_b5 = ConvBlock(mid_ch, dirate=2)

    conv_b4_d = ConvBlock(mid_ch, dirate=1)
    upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b3_d = ConvBlock(mid_ch, dirate=1)
    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b2_d = ConvBlock(mid_ch, dirate=1)
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b1_d = ConvBlock(out_ch, dirate=1)
    upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    hx = inputs
    hxin = conv_b0(hx)

    hx1 = conv_b1(hxin)
    hx = pool1(hx1)

    hx2 = conv_b2(hx)
    hx = pool2(hx2)

    hx3 = conv_b3(hx)
    hx = pool3(hx3)

    hx4 = conv_b4(hx)

    hx5 = conv_b5(hx4)

    hx4d = conv_b4_d(tf.concat([hx5, hx4], axis=3))
    hx4dup = upsample_3(hx4d)

    hx3d = conv_b3_d(tf.concat([hx4dup, hx3], axis=3))
    hx3dup = upsample_2(hx3d)

    hx2d =  conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
    hx2dup = upsample_1(hx2d)

    hx1d = conv_b1_d(tf.concat([hx2dup, hx1], axis=3))
    
    return hx1d + hxin


def RSU4(inputs, mid_ch=12, out_ch=3):
    conv_b0 = ConvBlock(out_ch, dirate=1)

    conv_b1 = ConvBlock(mid_ch, dirate=1)
    pool1   = MaxPool2D(2, strides=(2, 2))

    conv_b2 = ConvBlock(mid_ch, dirate=1)
    pool2   = MaxPool2D(2, strides=(2, 2))

    conv_b3 = ConvBlock(mid_ch, dirate=1)
    pool3   = MaxPool2D(2, strides=(2, 2))

    conv_b4 = ConvBlock(mid_ch, dirate=2)

    conv_b3_d = ConvBlock(mid_ch, dirate=1)
    upsample_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b2_d = ConvBlock(mid_ch, dirate=1)
    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    conv_b1_d = ConvBlock(out_ch, dirate=1)
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    hx = inputs
    hxin = conv_b0(hx)

    hx1 = conv_b1(hxin)
    hx = pool1(hx1)

    hx2 = conv_b2(hx)
    hx = pool2(hx2)

    hx3 = conv_b3(hx)

    hx4 = conv_b4(hx3)

    hx3d = conv_b3_d(tf.concat([hx4, hx3], axis=3))
    hx3dup = upsample_2(hx3d)

    hx2d =  conv_b2_d(tf.concat([hx3dup, hx2], axis=3))
    hx2dup = upsample_1(hx2d)

    hx1d = conv_b1_d(tf.concat([hx2dup, hx1], axis=3))
    
    return hx1d + hxin

def RSU4F(inputs, mid_ch=12, out_ch=3):
    conv_b0 = ConvBlock(out_ch, dirate=1)
    conv_b1 = ConvBlock(mid_ch, dirate=1)
    conv_b2 = ConvBlock(mid_ch, dirate=2)
    conv_b3 = ConvBlock(mid_ch, dirate=4)
    conv_b4 = ConvBlock(mid_ch, dirate=8)
    conv_b3_d = ConvBlock(mid_ch, dirate=4)
    conv_b2_d = ConvBlock(mid_ch, dirate=2)
    conv_b1_d = ConvBlock(out_ch, dirate=1)

    hx = inputs
    hxin = conv_b0(hx)
    
    hx1 = conv_b1(hxin)
    hx2 = conv_b2(hx1)
    hx3 = conv_b3(hx2)
    hx4 = conv_b4(hx3)
    hx3d = conv_b3_d(tf.concat([hx4, hx3], axis=3))
    hx2d = conv_b2_d(tf.concat([hx3d, hx2], axis=3))
    hx1d = conv_b1_d(tf.concat([hx2d, hx1], axis=3))
    return hx1d + hxin

def U2NET(inputs, out_ch=1):
    stage1 = RSU7(32, 64)
    pool12 = MaxPool2D((2, 2), 2)

    stage2 = RSU6(32, 128)
    pool23 = MaxPool2D((2, 2), 2)

    stage3 = RSU5(64, 256)
    pool34 = MaxPool2D((2, 2), 2)

    stage4 = RSU4(128, 512)
    pool45 = MaxPool2D((2, 2), 2)

    stage5 = RSU4F(256, 512)
    pool56 = MaxPool2D((2, 2), 2)

    stage6 = RSU4F(256, 512)

    stage5d = RSU4F(256, 512)
    stage4d = RSU4(128, 256)
    stage3d = RSU5(64, 128)
    stage2d = RSU6(32, 64)
    stage1d = RSU7(16, 64)

    side1 = Conv2D(out_ch, (3, 3), padding='same')
    side2 = Conv2D(out_ch, (3, 3), padding='same')
    side3 = Conv2D(out_ch, (3, 3), padding='same')
    side4 = Conv2D(out_ch, (3, 3), padding='same')
    side5 = Conv2D(out_ch, (3, 3), padding='same')
    side6 = Conv2D(out_ch, (3, 3), padding='same')

    upsample_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    upsample_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    upsample_4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    upsample_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    upsample_6 = UpSampling2D(size=(2, 2), interpolation='bilinear')

    upsample_out_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')
    upsample_out_3 = UpSampling2D(size=(4, 4), interpolation='bilinear')
    upsample_out_4 = UpSampling2D(size=(8, 8), interpolation='bilinear')
    upsample_out_5 = UpSampling2D(size=(16, 16), interpolation='bilinear')
    upsample_out_6 = UpSampling2D(size=(32, 32), interpolation='bilinear')

    outconv = Conv2D(out_ch, (1, 1), padding='same')

    hx = inputs

    hx1 = stage1(hx)
    hx = pool12(hx1)

    hx2 = stage2(hx)
    hx = pool23(hx2)

    hx3 = stage3(hx)
    hx = pool34(hx3)

    hx4 = stage4(hx)
    hx = pool45(hx4)

    hx5 = stage5(hx)
    hx = pool56(hx5)

    hx6 = stage6(hx)
    hx6up = upsample_6(hx6)
    side6 = upsample_out_6(side6(hx6))

    hx5d = stage5d(tf.concat([hx6up, hx5], axis=3))
    hx5dup = upsample_5(hx5d)
    side5 = upsample_out_5(side5(hx5d))

    hx4d = stage4d(tf.concat([hx5dup, hx4], axis=3))
    hx4dup = upsample_4(hx4d)
    side4 = upsample_out_4(side4(hx4d))

    hx3d = stage3d(tf.concat([hx4dup, hx3], axis=3))
    hx3dup = upsample_3(hx3d)
    side3 = upsample_out_3(side3(hx3d))

    hx2d = stage2d(tf.concat([hx3dup, hx2], axis=3))
    hx2dup = upsample_2(hx2d)
    side2 = upsample_out_2(side2(hx2d))

    hx1d = stage1d(tf.concat([hx2dup, hx1], axis=3))
    side1 = side1(hx1d)

    fused_output = outconv(tf.concat([side1, side2, side3, side4, side5, side6], axis=3))

    sig = keras.activations.sigmoid
    return tf.stack([sig(fused_output), sig(side1), sig(side2), sig(side3), sig(side4), sig(side5), sig(side6)])
