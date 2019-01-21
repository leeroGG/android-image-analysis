package com.smartahc.android.camera

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Looper
import android.os.MessageQueue
import android.provider.MediaStore
import android.support.v4.content.FileProvider
import android.support.v7.app.AppCompatActivity
import android.support.v7.app.AppCompatDelegate
import android.util.Log
import android.view.View
import android.widget.Toast
import com.bumptech.glide.Glide
import com.bumptech.glide.load.DataSource
import com.bumptech.glide.load.engine.GlideException
import com.bumptech.glide.request.RequestListener
import com.bumptech.glide.request.target.Target
import com.smartahc.android.camera.ternsorflow.Classifier
import com.smartahc.android.camera.ternsorflow.TensorFlowImageClassifier
import kotlinx.android.synthetic.main.activity_image_analysis.*
import java.io.File
import java.io.IOException
import java.util.concurrent.Executor
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.ThreadFactory


class ImageAnalysisActivity : AppCompatActivity(), View.OnClickListener {

    private val TAKE_PHOTO_REQUEST_CODE = 120
    private val PICTURE_REQUEST_CODE = 911

    private val CURRENT_TAKE_PHOTO_URI = "currentTakePhotoUri"

    private val INPUT_SIZE = 224
    private val IMAGE_MEAN = 117
    private val IMAGE_STD = 1f
    private val INPUT_NAME = "input"
    private val OUTPUT_NAME = "output"
    private val MODEL_FILE = "file:///android_asset/model/tensorflow_inception_graph.pb"
    private val LABEL_FILE = "file:///android_asset/model/imagenet_comp_graph_label_strings.txt"

    private var executor: Executor? = null
    private var currentTakePhotoUri: Uri? = null
    private var classifier: Classifier? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image_analysis)

        ivChoose.setOnClickListener(this)
        ivTakePhoto.setOnClickListener(this)

        // 避免耗时任务占用 CPU 时间片造成UI绘制卡顿，提升启动页面加载速度
        Looper.myQueue().addIdleHandler(idleHandler)
    }

    override fun onSaveInstanceState(outState: Bundle?) {
        outState?.putParcelable(CURRENT_TAKE_PHOTO_URI, currentTakePhotoUri)
        super.onSaveInstanceState(outState)
    }

    override fun onRestoreInstanceState(savedInstanceState: Bundle?) {
        super.onRestoreInstanceState(savedInstanceState)
        if (savedInstanceState != null) {
            currentTakePhotoUri = savedInstanceState.getParcelable(CURRENT_TAKE_PHOTO_URI)
        }
    }

    /**
     * 主线程消息队列空闲时（视图第一帧绘制完成时）处理耗时事件
     */
    private var idleHandler: MessageQueue.IdleHandler = MessageQueue.IdleHandler {
        if (classifier == null) {
            // 创建 Classifier
            classifier = TensorFlowImageClassifier.create(this@ImageAnalysisActivity.assets,
                    MODEL_FILE, LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME)
        }

        // 初始化线程池
        executor = ScheduledThreadPoolExecutor(1, ThreadFactory { r ->
            val thread = Thread(r)
            thread.isDaemon = true
            thread.name = "ThreadPool-ImageClassifier"
            thread
        })

        false
    }

    override fun onClick(view: View) {
        when (view.id) {
            R.id.ivChoose -> choosePicture()
            R.id.ivTakePhoto -> takePhoto()
            else -> {
            }
        }
    }

    /**
     * 选择一张图片并裁剪获得一个小图
     */
    private fun choosePicture() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        startActivityForResult(intent, PICTURE_REQUEST_CODE)
    }

    /**
     * 使用系统相机拍照
     */
    private fun takePhoto() {
        openSystemCamera()
    }

    /**
     * 打开系统相机
     */
    private fun openSystemCamera() {
        //调用系统相机
        val takePhotoIntent = Intent()
        takePhotoIntent.action = MediaStore.ACTION_IMAGE_CAPTURE

        //这句作用是如果没有相机则该应用不会闪退，要是不加这句则当系统没有相机应用的时候该应用会闪退
        if (takePhotoIntent.resolveActivity(packageManager) == null) {
            Toast.makeText(this, "当前系统没有可用的相机应用", Toast.LENGTH_SHORT).show()
            return
        }

        val fileName = "TF_" + System.currentTimeMillis() + ".jpg"
        val photoFile = File(FileUtil.getPhotoCacheFolder(), fileName)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            //通过FileProvider创建一个content类型的Uri
            currentTakePhotoUri = FileProvider.getUriForFile(this, "com.smartahc.android.camera.file", photoFile)
            //对目标应用临时授权该 Uri 所代表的文件
            takePhotoIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } else {
            currentTakePhotoUri = Uri.fromFile(photoFile)
        }

        //将拍照结果保存至 outputFile 的Uri中，不保留在相册中
        takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, currentTakePhotoUri)
        startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                PICTURE_REQUEST_CODE -> // 处理选择的图片
                    handleInputPhoto(data!!.data)
                TAKE_PHOTO_REQUEST_CODE -> // 如果拍照成功，加载图片并识别
                    handleInputPhoto(currentTakePhotoUri)
            }
        }
    }

    /**
     * 处理图片
     * @param imageUri
     */
    private fun handleInputPhoto(imageUri: Uri?) {
        // 加载图片
        Glide.with(this@ImageAnalysisActivity).asBitmap().listener(object : RequestListener<Bitmap> {

            override fun onLoadFailed(e: GlideException?, model: Any, target: Target<Bitmap>, isFirstResource: Boolean): Boolean {
                Toast.makeText(this@ImageAnalysisActivity, "图片加载失败", Toast.LENGTH_SHORT).show()
                return false
            }

            override fun onResourceReady(resource: Bitmap, model: Any, target: Target<Bitmap>, dataSource: DataSource, isFirstResource: Boolean): Boolean {
                startImageClassifier(resource)
                return false
            }
        }).load(imageUri).into(ivPicture)
        tvInfo.text = "Processing..."
    }

    /**
     * 开始图片识别匹配
     * @param bitmap
     */
    private fun startImageClassifier(bitmap: Bitmap) {
        executor?.execute{
            try {
                Log.i("111", Thread.currentThread().name + " startImageClassifier")
                val croppedBitmap = getScaleBitmap(bitmap, INPUT_SIZE)

                val results = classifier?.recognizeImage(croppedBitmap)
                Log.i("111", "startImageClassifier results: $results")
                runOnUiThread { tvInfo.text = String.format("results: %s", results) }
            } catch (e: IOException) {
                Log.e("111", "startImageClassifier getScaleBitmap " + e.message)
                e.printStackTrace()
            }
        }
    }


    /**
     * 对图片进行缩放
     * @param bitmap
     * @param size
     * @return
     * @throws IOException
     */
    @Throws(IOException::class)
    private fun getScaleBitmap(bitmap: Bitmap, size: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val scaleWidth = size.toFloat() / width
        val scaleHeight = size.toFloat() / height
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
    }
}
