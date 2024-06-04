#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"


#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];



#include "all.h"
#include <stdio.h>
#include <stdlib.h>


#include <math.h>
#include <float.h>
#include <stdint.h> // For int8_t

#include "quirc.h"

//===================================Preprocess===============================================

/************** interpolation constants and tables ***************/
#define ICV_WARP_SHIFT          10
#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
#define ICV_WARP_MUL_ONE_8U(x)  ((x) << ICV_WARP_SHIFT)
#define ICV_WARP_DESCALE_8U(x)  CV_DESCALE((x), ICV_WARP_SHIFT*2)
/****************************************************************************************\
*                                         Resize                                         *
\****************************************************************************************/

typedef struct CvResizeAlpha
{
    int idx;
    union
    {
        float alpha;
        int ialpha;
    };
}
CvResizeAlpha;

typedef unsigned short ushort;

#define  ICV_DEF_RESIZE_BILINEAR_FUNC( flavor, arrtype, worktype, alpha_field,  \
                                       mul_one_macro, descale_macro )           \
static CvStatus CV_STDCALL                                                      \
icvResize_Bilinear_##flavor##_CnR( const arrtype* src, int srcstep, CvSize ssize,\
                                   arrtype* dst, int dststep, CvSize dsize,     \
                                   int cn, int xmax,                            \
                                   const CvResizeAlpha* xofs,                   \
                                   const CvResizeAlpha* yofs,                   \
                                   worktype* buf0, worktype* buf1 )             \
{                                                                               \
    int prev_sy0 = -1, prev_sy1 = -1;                                           \
    int k, dx, dy;                                                              \
                                                                                \
    srcstep /= sizeof(src[0]);                                                  \
    dststep /= sizeof(dst[0]);                                                  \
    dsize.width *= cn;                                                          \
    xmax *= cn;                                                                 \
                                                                                \
    for( dy = 0; dy < dsize.height; dy++, dst += dststep )                      \
    {                                                                           \
        worktype fy = yofs[dy].alpha_field, *swap_t;                            \
        int sy0 = yofs[dy].idx, sy1 = sy0 + (fy > 0 && sy0 < ssize.height-1);   \
                                                                                \
        if( sy0 == prev_sy0 && sy1 == prev_sy1 )                                \
            k = 2;                                                              \
        else if( sy0 == prev_sy1 )                                              \
        {                                                                       \
            CV_SWAP( buf0, buf1, swap_t );                                      \
            k = 1;                                                              \
        }                                                                       \
        else                                                                    \
            k = 0;                                                              \
                                                                                \
        for( ; k < 2; k++ )                                                     \
        {                                                                       \
            worktype* _buf = k == 0 ? buf0 : buf1;                              \
            const arrtype* _src;                                                \
            int sy = k == 0 ? sy0 : sy1;                                        \
            if( k == 1 && sy1 == sy0 )                                          \
            {                                                                   \
                memcpy( buf1, buf0, dsize.width*sizeof(buf0[0]) );              \
                continue;                                                       \
            }                                                                   \
                                                                                \
            _src = src + sy*srcstep;                                            \
            for( dx = 0; dx < xmax; dx++ )                                      \
            {                                                                   \
                int sx = xofs[dx].idx;                                          \
                worktype fx = xofs[dx].alpha_field;                             \
                worktype t = _src[sx];                                          \
                _buf[dx] = mul_one_macro(t) + fx*(_src[sx+cn] - t);             \
            }                                                                   \
                                                                                \
            for( ; dx < dsize.width; dx++ )                                     \
                _buf[dx] = mul_one_macro(_src[xofs[dx].idx]);                   \
        }                                                                       \
                                                                                \
        prev_sy0 = sy0;                                                         \
        prev_sy1 = sy1;                                                         \
                                                                                \
        if( sy0 == sy1 )                                                        \
            for( dx = 0; dx < dsize.width; dx++ )                               \
                dst[dx] = (arrtype)descale_macro( mul_one_macro(buf0[dx]));     \
        else                                                                    \
            for( dx = 0; dx < dsize.width; dx++ )                               \
                dst[dx] = (arrtype)descale_macro( mul_one_macro(buf0[dx]) +     \
                                                  fy*(buf1[dx] - buf0[dx]));    \
    }                                                                           \
                                                                                \
    return CV_OK;                                                               \
}



ICV_DEF_RESIZE_BILINEAR_FUNC( 8u, uchar, int, ialpha,
                              ICV_WARP_MUL_ONE_8U, ICV_WARP_DESCALE_8U )
ICV_DEF_RESIZE_BILINEAR_FUNC( 16u, ushort, float, alpha, CV_NOP, cvRound )
ICV_DEF_RESIZE_BILINEAR_FUNC( 32f, float, float, alpha, CV_NOP, CV_NOP )




static void icvInitResizeTab( CvFuncTable* bilin_tab,
                              CvFuncTable* bicube_tab,
                              CvFuncTable* areafast_tab,
                              CvFuncTable* area_tab )
{
    bilin_tab->fn_2d[CV_8U] = (void*)icvResize_Bilinear_8u_CnR;
    bilin_tab->fn_2d[CV_16U] = (void*)icvResize_Bilinear_16u_CnR;
    bilin_tab->fn_2d[CV_32F] = (void*)icvResize_Bilinear_32f_CnR;

}


typedef CvStatus (CV_STDCALL * CvResizeBilinearFunc)
                    ( const void* src, int srcstep, CvSize ssize,
                      void* dst, int dststep, CvSize dsize,
                      int cn, int xmax, const CvResizeAlpha* xofs,
                      const CvResizeAlpha* yofs, float* buf0, float* buf1 );


void simplifiedCvCopy(const CvMat* src, CvMat* dst) {

    if (!CV_IS_MAT(src) || !CV_IS_MAT(dst) ||
        !CV_ARE_TYPES_EQ(src, dst) ||
        !CV_ARE_SIZES_EQ(src, dst)) {
        // 错误处理：不匹配的类型或尺寸
        fprintf(stderr, "Source and destination must have the same type and size.\n");
        return;
    }

    // 计算每行需要复制的字节数
    size_t bytesPerRow = src->cols * CV_ELEM_SIZE(src->type);

    // 逐行复制数据
    for (int i = 0; i < src->rows; ++i) {
        memcpy(dst->data.ptr + i * dst->step, 
               src->data.ptr + i * src->step, 
               bytesPerRow);
    }
}

void simplifiedCvGetMat(const CvArr* array, CvMat* mat, int* pCOI) {
    assert(array != NULL && mat != NULL);

    CvMat* src = (CvMat*)array;

    if (!CV_IS_MAT_HDR(src) || !src->data.ptr) {
        fprintf(stderr, "Input is not a valid CvMat or has NULL data pointer.\n");
        return; // 可以在这里处理错误
    }
    *mat = *src;
    if (pCOI) *pCOI = 0;
}

CV_IMPL void
cvResize( const CvArr* srcarr, CvArr* dstarr, int method )
{
    static CvFuncTable bilin_tab, bicube_tab, areafast_tab, area_tab;
    static int inittab = 0;
    void* temp_buf = 0;
    __BEGIN__;
    
    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize ssize, dsize;
    float scale_x, scale_y;
    int k, sx, sy, dx, dy;
    int type, depth, cn;
    
    simplifiedCvGetMat(srcarr, &srcstub, NULL);
    simplifiedCvGetMat(dstarr, &dststub, NULL);
    
    src = &srcstub;
    dst = &dststub;

    if( CV_ARE_SIZES_EQ( src, dst ))
        simplifiedCvCopy( src, dst );

    if( !inittab )
    {
        icvInitResizeTab( &bilin_tab, &bicube_tab, &areafast_tab, &area_tab );
        inittab = 1;
    }

    ssize = cvGetMatSize( src );
    dsize = cvGetMatSize( dst );
    type = CV_MAT_TYPE(src->type);
    depth = CV_MAT_DEPTH(type);
    cn = CV_MAT_CN(type);
    scale_x = (float)ssize.width/dsize.width;
    scale_y = (float)ssize.height/dsize.height;

    if( method == CV_INTER_LINEAR )
    {
        float inv_scale_x = (float)dsize.width/ssize.width;
        float inv_scale_y = (float)dsize.height/ssize.height;
        int xmax = dsize.width, width = dsize.width*cn, buf_size;
        float *buf0, *buf1;
        CvResizeAlpha *xofs, *yofs;
        int area_mode = method == CV_INTER_AREA;
        float fx, fy;
        CvResizeBilinearFunc func = (CvResizeBilinearFunc)bilin_tab.fn_2d[depth];

        buf_size = width*2*sizeof(float) + (width + dsize.height)*sizeof(CvResizeAlpha);
        // Replaced cvStackAlloc and cvAlloc with malloc
        buf0 = (float*)malloc(buf_size);
        if (!buf0) {
            // Handle memory allocation failure
            fprintf(stderr, "Failed to allocate memory for buffers.\n");
            return;
        }
        temp_buf = buf0; // Keep track of the allocated buffer for later free
        buf1 = buf0 + width;
        xofs = (CvResizeAlpha*)(buf1 + width);
        yofs = xofs + width;

        for( dx = 0; dx < dsize.width; dx++ )
        {
            if( !area_mode )
            {
                fx = (float)((dx+0.5)*scale_x - 0.5);
                // sx = cvFloor(fx);
                sx = (int)floor(fx);
                fx -= sx;
            }
            else
            {
                sx = (int)floor(dx*scale_x);
                fx = (dx+1) - (sx+1)*inv_scale_x;
                fx = fx <= 0 ? 0.f : fx - (int)floor(fx);
            }

            if( sx < 0 )
                fx = 0, sx = 0;

            if( sx >= ssize.width-1 )
            {
                fx = 0, sx = ssize.width-1;
                if( xmax >= dsize.width )
                    xmax = dx;
            }

            if( depth != 0 )
                for( k = 0, sx *= cn; k < cn; k++ )
                    xofs[dx*cn + k].idx = sx + k, xofs[dx*cn + k].alpha = fx;
            else
                for( k = 0, sx *= cn; k < cn; k++ )
                    xofs[dx*cn + k].idx = sx + k,
                    xofs[dx*cn + k].ialpha = CV_FLT_TO_FIX(fx, ICV_WARP_SHIFT);
        }

        for( dy = 0; dy < dsize.height; dy++ )
        {
            if( !area_mode )
            {
                fy = (float)((dy+0.5)*scale_y - 0.5);
                sy = (int)floor(fy);
                fy -= sy;
                if( sy < 0 )
                    sy = 0, fy = 0;
            }
            else
            {
                sy = (int)floor(dy*scale_y);
                fy = (dy+1) - (sy+1)*inv_scale_y;
                fy = fy <= 0 ? 0.f : fy - (int)floor(fy);
            }

            yofs[dy].idx = sy;
            if( depth != CV_8U )
                yofs[dy].alpha = fy;
            else
                yofs[dy].ialpha = CV_FLT_TO_FIX(fy, ICV_WARP_SHIFT);
        }

        func( src->data.ptr, src->step, ssize, dst->data.ptr,
                dst->step, dsize, cn, xmax, xofs, yofs, buf0, buf1 );
    }

    __END__;

    free(temp_buf); // Replace cvFree with free
}



/****************************************************************************************\
*                                     BGRtoRGB & Normalize                               *
\****************************************************************************************/
void convertBGRtoRGB(CvMat* image) {
    // CV_ASSERT(image != NULL && image->data.ptr != NULL);
    int channels = CV_MAT_CN(image->type);
    int width = image->cols;
    int height = image->rows;
    int step = image->step ? image->step : CV_STUB_STEP; // 使用CV_STUB_STEP处理连续性

    if (channels != 3) {
        fprintf(stderr, "convertBGRtoRGB: 图像不是3通道的。\n");
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uchar* pixel = image->data.ptr + y * step + x * 3;
            // 交换第一个和第三个通道的值（B和R）
            uchar temp = pixel[0];
            pixel[0] = pixel[2];
            pixel[2] = temp;
        }
    }
}


typedef struct {
    int width;
    int height;
    int channels;
    int batchSize;
    int8_t* data;
} INT8Image;



void cvMatToINT8ImageAndNormalize(CvMat* src, INT8Image* dst) {
    const float mean[3] = {128.0f, 128.0f, 128.0f}; // RGB顺序

    dst->width = src->cols;
    dst->height = src->rows;
    dst->channels = CV_MAT_CN(src->type);
    dst->data = (int8_t*)malloc(dst->width * dst->height * dst->channels * dst->batchSize * sizeof(int8_t));
   
    int step = src->step ? src->step : src->cols * dst->channels * sizeof(uchar);

    for (int c = 0; c < dst->channels; c++) {
        for (int y = 0; y < dst->height; y++) {
            for (int x = 0; x < dst->width; x++) {
                uchar* srcPixel = src->data.ptr + y * step + x * dst->channels;
                int8_t* dstPixel = dst->data + (c * dst->width * dst->height) + y * dst->width + x;
                float value = (srcPixel[c] - mean[c]) / 2;
                dstPixel[0] = (int8_t)lrintf(value); // 使用 lrintf 并强制转换为 int8_t
            }
        }
    }
}


//===================================Postprocess===============================================

#define GRID_NUM 8
#define CELL_SIZE (1.0 / GRID_NUM)
#define MAX_BOXES 100 // 假设最多100个框
#define MAX_RESULTS 100 // 假设最多有100个检测结果

typedef struct {
    float x1, y1, x2, y2; // 框的坐标
} Box;

typedef struct {
    float prob; // Detection probability
} BoxInfo;

// 用于排序的结构
typedef struct {
    int index;
    float score;
} ScoreIndex;


typedef struct {
    int x1, y1, x2, y2; // 使用整数坐标
    char* class_name;   // 类别名称
    float prob;         // 置信度
} DetectionResult;


int boxCount = 0;
int resultCount = 0;
char* VOC_CLASSES[] = {"qr_code"};

Box boxes[MAX_BOXES];
BoxInfo boxInfos[MAX_BOXES];
DetectionResult results[MAX_RESULTS];


// 用于qsort的比较函数，降序排序
int compare(const void *a, const void *b) {
    ScoreIndex *boxA = (ScoreIndex *)a;
    ScoreIndex *boxB = (ScoreIndex *)b;
    float diff = boxB->score - boxA->score;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

float IoU(Box a, Box b) {
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    float intersectionX1 = fmax(a.x1, b.x1);
    float intersectionY1 = fmax(a.y1, b.y1);
    float intersectionX2 = fmin(a.x2, b.x2);
    float intersectionY2 = fmin(a.y2, b.y2);

    float intersectionArea = fmax(0, intersectionX2 - intersectionX1) * fmax(0, intersectionY2 - intersectionY1);
    float unionArea = areaA + areaB - intersectionArea;

    return intersectionArea / unionArea;
}

int* nms(Box* boxes, float* scores, int num_boxes, float threshold, int* num_keep) {
    ScoreIndex* si = (ScoreIndex*)malloc(num_boxes * sizeof(ScoreIndex));
    for (int i = 0; i < num_boxes; i++) {
        si[i].index = i;
        si[i].score = scores[i];
    }

    // 对分数进行降序排序
    qsort(si, num_boxes, sizeof(ScoreIndex), compare);

    int* keep = (int*)malloc(num_boxes * sizeof(int));
    int* suppressed = (int*)calloc(num_boxes, sizeof(int));
    *num_keep = 0;

    for (int i = 0; i < num_boxes; ++i) {
        if (suppressed[si[i].index] == 1) continue;

        keep[(*num_keep)++] = si[i].index;

        for (int j = i + 1; j < num_boxes; ++j) {
            if (suppressed[si[j].index] == 1) continue;

            if (IoU(boxes[si[i].index], boxes[si[j].index]) > threshold) {
                suppressed[si[j].index] = 1;
            }
        }
    }

    free(si);
    free(suppressed);
    return keep;
}


void processDetections(int* keep, int num_keep, int IMAGE_WIDTH, int IMAGE_HEIGHT) {
    resultCount = 0;
    for (int i = 0; i < num_keep; ++i) {
        int idx = keep[i];
        results[i].x1 = (int)(boxes[idx].x1 * IMAGE_WIDTH);
        results[i].y1 = (int)(boxes[idx].y1 * IMAGE_HEIGHT);
        results[i].x2 = (int)(boxes[idx].x2 * IMAGE_WIDTH);
        results[i].y2 = (int)(boxes[idx].y2 * IMAGE_HEIGHT);
        results[i].class_name = "qr_code";
        results[i].prob = boxInfos[idx].prob;
        resultCount++;
    }
}

//===============================Main==================================================


static void load_bin_file(const char *file, int8_t **ptr, uint64_t *size) 
{
    FILE *fp = fopen(file, "rb");
	if (fp == NULL){
		printf("open file failed, check the path!\n");
	}

    fseek(fp, 0 ,SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0 ,SEEK_SET);
    *ptr = (int8_t *)malloc(*size);
    fread(*ptr, *size, 1, fp);
    fclose(fp);
}

static void save_bin_file(const char *file, int8_t *ptr, int32_t size)
{
	FILE *fp = fopen(file, "ab+");

	fwrite(ptr, size, 1, fp);

	fclose(fp);
}

int thinker_task_test(int loop_count, char *argv[])
{
	int i, j, k;
	int32_t use_psram_size = 0;
	int32_t use_share_size = 0;

    memset(g_psram_buf, 0, PSRAM_SIZE);
    memset(g_share_buf, 0, SHARE_SIZE);

    int8_t *input_data = NULL;
    int8_t *res_data = NULL;
    uint64_t input_size = 0;
    uint64_t res_size = 0;

	char *input_file = argv[1];
	char *model_file = argv[2];
	char *output_file = argv[3];
	int32_t in_c = atoi(argv[4]);
	int32_t in_h = atoi(argv[5]);
	int32_t in_w = atoi(argv[6]);
	int32_t scale = atoi(argv[7]);

//=====================================================================================
    FILE* file = fopen(input_file, "rb");
    if (!file) {
        printf("Unable to open file %s\n", input_file);
        return -1;
    }

    // 读取图像尺寸和通道数
    int width, height, channels;
    fread(&height, sizeof(int), 1, file);
    fread(&width, sizeof(int), 1, file);
    fread(&channels, sizeof(int), 1, file);

    // 根据读取的尺寸和通道数分配源图像数据空间
    CvSize srcSize = {width, height};
    int srcStep = width * channels * sizeof(uchar);
    uchar* srcData = (uchar*)malloc(height * srcStep);

    // 读取图像数据
    fread(srcData, sizeof(uchar), height * srcStep, file);
    fclose(file);

    // 创建目标图像数据
    CvSize dstSize = {64, 64};
    int dstStep = dstSize.width * channels * sizeof(uchar);
    uchar* dstData = (uchar*)malloc(dstSize.height * dstStep);

    // 初始化源和目标 CvMat 结构
    CvMat srcCvMat = cvMat(srcSize.height, srcSize.width, CV_8UC3, srcData);
    CvMat dstCvMat = cvMat(dstSize.height, dstSize.width, CV_8UC3, dstData);

    // 调用自定义的 cvResize 函数进行缩放操作
    cvResize(&srcCvMat, &dstCvMat, CV_INTER_LINEAR);
    convertBGRtoRGB(&dstCvMat);
    INT8Image INT8ImageDst;
    INT8ImageDst.batchSize = 1; // 单张图片或批量处理时调整此值
    cvMatToINT8ImageAndNormalize(&dstCvMat, &INT8ImageDst);

//=====================================================================================

	// load_bin_file(input_file, &input_data, &input_size);
    load_bin_file(model_file, &res_data, &res_size);

    tStatus ret = T_SUCCESS;
	ret = tInitialize();
	if (ret != T_SUCCESS) {
        printf("tInitialize failed, error code:%d\n", ret);
		return ret;
    }

	int num_memory = 0;
	tMemory memory_list[7];
	ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res_data, res_size);
    if (ret != T_SUCCESS) {
        printf("tGetMemoryPlan failed, error code:%d\n", ret);
		return ret;
    }

	for(i = 0; i < num_memory; i++)
	{
		int mem_size = memory_list[i].size_;
		if (memory_list[i].dptr_ == 0)
		{
			if (1 == memory_list[i].dev_type_ || 3 == memory_list[i].dev_type_)
			{
				memory_list[i].dptr_ = (uint64_t)(g_psram_buf + use_psram_size);
				use_psram_size += (mem_size+63)&(~63);
			}
			else if (2 == memory_list[i].dev_type_)
			{
				memory_list[i].dptr_ = (uint64_t)(g_share_buf + use_share_size);
				use_share_size += (mem_size+63)&(~63);
			}
		}
	}

    tModelHandle model_hdl;   //typedef uint64_t
    ret = tModelInit(&model_hdl, (int8_t*)res_data, res_size, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tInitModel failed, error code:%d\n", ret);
		return ret;
    }
	else{
		printf("init model successful!\n");
	}

    tExecHandle hdl;
    ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tCreateExecutor failed, error code:%d\n", ret);
		return ret;
    }
	else{
		printf("create executor successful!\n");
	}

  	tData input; 
	input.dptr_ = (char*)INT8ImageDst.data;
	input.dtype_ = Int8;
	input.scale_ = scale;
	input.shape_.ndim_ = 4;
	input.shape_.dims_[0] = 1;
	input.shape_.dims_[1] = in_c;
    input.shape_.dims_[2] = in_h;
    input.shape_.dims_[3] = in_w;


	uint32_t clk = 0;
	for(i = 0; i < loop_count; i++)
	{
		ret = tSetInput(hdl, 0, &input);
		if (ret != T_SUCCESS) {
			printf("tSetInput failed, error coe:%d\n", ret);
			return ret;
		}

		ret = tForward(hdl);
		if (ret != T_SUCCESS) {
			printf("tForward failed, error code:%d\n", ret);
			return ret;
		}
		else{
			printf("forward successful!\n");
		}

		tData output[5];
		int getoutputcount = tGetOutputCount(model_hdl);

		for(j = 0; j < getoutputcount; j++)
		{
			ret = tGetOutput(hdl, j, &output[j]);
			if (ret != T_SUCCESS) {
				printf("tGetOutput_%d failed, error code: %d\n", j, ret);
				return ret;
			}
		}

		int8_t *output_data = (int8_t *)output[0].dptr_;
		int output_length = output[0].shape_.dims_[1];


		// 创建一个浮点数组来存储转换后的结果
		float *pred = (float *)malloc(640 * sizeof(float));
		if (!pred) {
			printf("Memory allocation failed\n");
			return -1;
		}

		// 转换数据
		for (int index = 0; index < 640; ++index) {
			pred[index] = output_data[index] / 128.0;
		}


		// 已知的尺寸信息
		int batchSize = 1;
		int rows = 8;
		int cols = 8;
		int channels = 10;


		// contain1 和 contain2 的计算
		float* contain1 = (float*)malloc(rows * cols * sizeof(float));
		float* contain2 = (float*)malloc(rows * cols * sizeof(float));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				contain1[r * cols + c] = pred[r * cols * channels + c * channels + 4]; // 第5个通道
				contain2[r * cols + c] = pred[r * cols * channels + c * channels + 9]; // 第10个通道
			}
		}


		// contain 的合并
		float* contain = (float*)malloc(rows * cols * 2 * sizeof(float));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				contain[r * cols * 2 + c * 2 + 0] = contain1[r * cols + c];
				contain[r * cols * 2 + c * 2 + 1] = contain2[r * cols + c];
			}
		}

		// mask1 和 mask2 的计算
		int* mask = (int*)malloc(rows * cols * 2 * sizeof(int));
		float max_value = -FLT_MAX;
		for (int i = 0; i < rows * cols * 2; ++i) {
			if (contain[i] > max_value) {
				max_value = contain[i];
			}
		}

		for (int i = 0; i < rows * cols * 2; ++i) {
			int mask1 = contain[i] > 0.1 ? 1 : 0;
			int mask2 = contain[i] == max_value ? 1 : 0;
			mask[i] = (mask1 || mask2) ? 1 : 0;
		}



        for (int i = 0; i < GRID_NUM; ++i) {
            for (int j = 0; j < GRID_NUM; ++j) {
                for (int b = 0; b < 2; ++b) {
                    if (boxCount >= MAX_BOXES) break; // Prevent array overflow

                    int index = i * GRID_NUM * 2 + j * 2 + b;
                    if (mask[index]) {
                        float cx = pred[(i * GRID_NUM * channels + j * channels + b * 5) + 0];
                        float cy = pred[(i * GRID_NUM * channels + j * channels + b * 5) + 1];
                        float w = pred[(i * GRID_NUM * channels + j * channels + b * 5) + 2];
                        float h = pred[(i * GRID_NUM * channels + j * channels + b * 5) + 3];
                        float prob = pred[i * GRID_NUM * channels + j * channels + b * 5 + 4];

                        float x = j * CELL_SIZE + cx * CELL_SIZE - w / 2.0;
                        float y = i * CELL_SIZE + cy * CELL_SIZE - h / 2.0;

                        boxes[boxCount].x1 = x;
                        boxes[boxCount].y1 = y;
                        boxes[boxCount].x2 = x + w;
                        boxes[boxCount].y2 = y + h;

                        if (prob > 0.1) {
                            boxInfos[boxCount].prob = prob;
                            ++boxCount;
						}
					}
				}
			}
		}
        
		if (boxCount == 0) {
			printf("No boxes detected.\n");
			Box boxes[1];
			memset(boxes, 0, sizeof(Box));
			float probs[1] = {0};
			int cls_indexs[1] = {0};

		} else {
			float* scores = (float*)malloc(boxCount * sizeof(float));
			for (int i = 0; i < boxCount; ++i) {
				scores[i] = boxInfos[i].prob;
			}

			int num_keep = 0;
			int* keep = nms(boxes, scores, boxCount, 0.1, &num_keep);
			processDetections(keep, num_keep,width,height);


            for (int i = 0; i < resultCount; ++i) {
                // 打印检测框结果
                printf("Result %d: [%d, %d, %d, %d], Class: %s, Prob: %.4f\n",
                    i, results[i].x1, results[i].y1, results[i].x2, results[i].y2,
                    results[i].class_name, results[i].prob);

                // 扩展边界框，增加10%的边界
                int padding = 0.1 * fmin(results[i].x2 - results[i].x1, results[i].y2 - results[i].y1); // 计算10%的边界余量
                int x1 = fmax(0, results[i].x1 - padding);
                int y1 = fmax(0, results[i].y1 - padding);
                int x2 = fmin(width, results[i].x2 + padding);
                int y2 = fmin(height, results[i].y2 + padding);

                // 裁剪区域对应的灰度图像
                uchar *grayData = malloc((y2 - y1) * (x2 - x1));
                if (!grayData) {
                    fprintf(stderr, "Memory allocation for grayscale image failed.\n");
                    continue;
                }

                // 将BGR转为灰度
                for (int y = y1; y < y2; y++) {
                    for (int x = x1; x < x2; x++) {
                        int idx = y * width + x;
                        uchar b = srcData[3 * idx];
                        uchar g = srcData[3 * idx + 1];
                        uchar r = srcData[3 * idx + 2];
                        grayData[(y - y1) * (x2 - x1) + (x - x1)] = (uchar)(0.299 * r + 0.587 * g + 0.114 * b);
                    }
                }

                // 使用Quirc解码
                struct quirc* qr = quirc_new();
                if (quirc_resize(qr, x2 - x1, y2 - y1) < 0 || !qr) {
                    fprintf(stderr, "Failed to initialize or resize quirc decoder.\n");
                    if (qr) quirc_destroy(qr);
                    free(grayData);
                    continue;
                }

                memcpy(quirc_begin(qr, NULL, NULL), grayData, (y2 - y1) * (x2 - x1));
                quirc_end(qr);

                int num_codes = quirc_count(qr);
                if (num_codes > 0) {
                    for (int j = 0; j < num_codes; j++) {
                        struct quirc_code code;
                        struct quirc_data data;
                        quirc_extract(qr, j, &code);
                        int decode_status = quirc_decode(&code, &data);
                        if (decode_status == QUIRC_SUCCESS) {
                            printf("Decoded QR code: %s\n", data.payload);
                        } else {
                            printf("QR code decode failed: %s\n", quirc_strerror(decode_status));
                        }
                    }
                } else {
                    printf("No QR code found or decode failed\n");
                }

                quirc_destroy(qr);
                free(grayData);
            }

			free(scores);
			free(keep);
		}
				
		free(contain1);
		free(contain2);
		free(contain);
		free(mask);
		free(pred);
	
		save_bin_file(output_file, output_data, output_length);

	}
	tUninitialize();
    return ret;
}


int main(int argc, char *argv[]) 
{
	if(argc < 8)
	{
		printf("commad:path of input file, path of model, path of output file, channel of input, height of input, width of input, QValue of input, loop num(opt)\n");
		return -1;
	}
	int loop_count = 1;
	if( argc == 9)
		loop_count = atoi(argv[8]);
	int32_t ret = thinker_task_test(loop_count, argv);

	return ret;
}
