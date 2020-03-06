import tensorflow as tf


class TransMatGen:
    def __init__(self):
        # if len(img_dims) != 5:
        #     print(img_dims, file=sys.stderr)
        #     raise ValueError('Must be 5 image volume dimensions')
        # else:
        #     self._img_dims = img_dims
        pass


    def flipMat(self, mb_size):
        flip_mat = tf.round(tf.random.uniform([mb_size, 2, 2], 0, 1))
        flip_mat = (flip_mat * 2) - 1
        flip_mat = flip_mat * tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])

        return flip_mat


    def rotMat(self, mb_size):
        thetas = tf.random.uniform([mb_size], -90, 90)
        thetas = thetas / 180 * 3.14159265359

        rot_00 = tf.math.cos(thetas)
        rot_01 = -tf.math.sin(thetas)
        rot_10 = tf.math.sin(thetas)
        rot_11 = tf.math.cos(thetas)
        rot_mat = tf.stack([[rot_00, rot_01], [rot_10, rot_11]])
        rot_mat = tf.transpose(rot_mat, [2, 0, 1])

        return rot_mat


    def scaleMat(self, mb_size):
        z = tf.random.uniform([mb_size], 0.75, 1.25)
        scale_mat = tf.tile(tf.eye(2)[tf.newaxis, :, :], [mb_size, 1, 1])
        scale_mat = (scale_mat * z[:, tf.newaxis, tf.newaxis])

        return scale_mat


    def shearMat(self, phi):
        phi = phi / 180 * np.pi
        phi = np.random.uniform(-phi, phi)

        p = False
        p = bool(np.random.binomial(1, 0.5))

        shear_mat = np.copy(self._ident_mat)

        if p:
            shear_mat[0, 1] = phi
        else:
            shear_mat[1, 0] = phi

        return shear_mat


    def transMatGen(self, mb_size):
        # trans_mat = self._ident_mat

        # if flip == None:
        #     pass
        # elif flip < 0 or flip > 1:
        #     raise ValueError("Flip probability out must be between 0 and 1")
        # else:
        #     trans_mat = np.matmul(trans_mat, self.flipMat(flip))
        #
        # if rot != None:
        #     trans_mat = np.matmul(trans_mat, self.rotMat(rot))
        #
        # if scale != None:
        #     trans_mat = np.matmul(trans_mat, self.scaleMat(scale))
        #
        # if shear != None:
        #     trans_mat = np.matmul(trans_mat, self.shearMat(shear))
        trans_mat = tf.matmul(self.flipMat(mb_size), tf.matmul(self.rotMat(mb_size), self.scaleMat(mb_size)))

        return trans_mat

if __name__ == "__main__":
    TestMatGen = TransMatGen(4)
    print(TestMatGen.flipMat())
    print(TestMatGen.rotMat())
    print(TestMatGen.scaleMat())
    print(TestMatGen.transMatGen())
