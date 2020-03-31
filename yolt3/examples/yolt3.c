#include "darknet.h"


// AVE:
// IN YOLOV3 DETECTION IS CALLED WITH ./DARKNET DETECT ..., WHICH CALLS
// TRAIN DETECTOR.C.  THEREFORE THIS FILE IS A HIGHLY MODIFIED VERSION 
// OF DETECTOR.C
// IF NBANDS > 3, CAN'T USE HSV RESCALING


// #include "network.h"
// #include "region_layer.h"
// #include "cost_layer.h"
// #include "utils.h"
// #include "parser.h"
// #include "box.h"
// #include "demo.h"
// #include "option_list.h"
//
// #ifdef OPENCV
// #include "opencv2/highgui/highgui_c.h"
// #endif

// for list parsing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
	
// AVE: Split string into list
// http://stackoverflow.com/questions/9210528/split-string-with-delimiters-in-c
char** str_split2(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

// // AVE:
// image rgb_from_multiband(image im)
// {
// 	//AVE: take first three bands from multiband
// 	int nbands_out = 3;
//     image im_out = make_image(im.w, im.h, nbands_out);
//     int x,y,k;
//     for(k = 0; k < nbands_out; ++k){
//         for(y = 0; y < im.h; ++y){
//             for(x = 0; x < im.w; ++x){
//                 float val = get_pixel(im, x,y,k);
//                 set_pixel(im_out, x, y, k, val);
//             }
//         }
//     }
// 	return im_out;
// }
//
// // AVE
// void save_image_ave(image im, char *out_dir, const char *name)
// {
//     char buff[256];
//     //char buff[256];
//     //sprintf(buff, "%s (%d)", name, windows);
// 	//sprintf(buff, "%s.png", name);
// 	fprintf(stderr, "out_name: %s/%s.png", out_dir, name);
//     sprintf(buff, "%s/%s.png", out_dir, name);
//     unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
//     int i,k;
//     for(k = 0; k < im.c; ++k){
//         for(i = 0; i < im.w*im.h; ++i){
//             data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
//         }
//     }
//     int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
//     free(data);
//     if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
// }

static float get_pixel2(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel2(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

// AVE:
image rgb_from_multiband2(image im, int w, int h)
{
	//AVE: take first three bands from multiband
	int nbands_out = 3;
    image im_out = make_image(im.w, im.h, nbands_out);
    int x,y,k;
    for(k = 0; k < nbands_out; ++k){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float val = get_pixel2(im, x,y,k);
                set_pixel2(im_out, x, y, k, val);
            }
        }
    }
	return im_out;
}
// AVE
void print_yolt_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

// AVE
void train_yolt3(char *cfgfile, char *weightfile, char *train_images, char *results_dir, int nbands, char *loss_file, int *gpus, int ngpus, int reset_seen)
{	
	
    srand(time(0));
    char *base = basecfg(cfgfile);
    fprintf(stderr, "basecfg: %s\n", base);
    float avg_loss = -1;

	////////
	// set loss file
	FILE *lossfile;
	//if(loss_file){
	//	lossfile = fopen(loss_file, "a");
	//}
	lossfile = fopen(loss_file, "a");
    if (lossfile == NULL) {
       fprintf(stderr, "Couldn't open lossfile for appending.\n");
       exit(0);
    }
	fprintf(lossfile, "%s,%s,%s,%s\n", "Batch_Num", "BatchSize", "N_Train_Ims", "Loss");
    fclose(lossfile);
    fprintf(stderr, "%s,%s,%s,%s\n", "Batch_Num", "BatchSize", "N_Train_Ims", "Loss");
	////////

	// set network
    fprintf(stderr, "%s\n", "Set Network");
    network **nets = calloc(ngpus, sizeof(network));
    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, reset_seen);
        // if(reset_seen) *nets[i].seen = 0;
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
	i = 0;  // *net.seen/imgs;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    fprintf(stderr, "\n\n\n\nNum images = %d,\ni= %d\n", imgs, i);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    //int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
	args.c = nbands;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 16;  // originally, 64

    pthread_t load_thread = load_data(args);
	fprintf(stderr, "N ims: %d\n", N);
	//int stop_count = net.max_batches;
	fprintf(stderr, "Num iters: %d\n", net->max_batches);
	// begin=clock();
	// clock_t time;
	// clock_t begin;
	// clock_t end;
    double start = what_time_is_it_now();
    double time;
    int count = 0;
	int j = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
		j += 1;
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 544; // originally, 608
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
		fprintf(stderr, "Batch Num: %d / %d\n", j, net->max_batches);
		
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        fprintf(stderr, "%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);

		// save loss to file
		if (i % 50 == 0){
            FILE *lossfile = fopen(loss_file, "a");
            fprintf(lossfile, "%d,%d,%d,%f\n", i, imgs, N, loss);
            fclose(lossfile);
			//fprintf(lossfile, "%d,%d,%d,%f\n", i, imgs, N, loss);
		}
		
        //if(i%5000==0 || (i < 1001 && i%250 == 0)){
        if(i%5000==0 || (i == 1000)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", results_dir, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", results_dir, base);
    save_weights(net, buff);

    fprintf(stderr, "Total Elapsed Training Time: %f Seconds\n", what_time_is_it_now() - start);
    // fprintf(stderr, "Total Elapsed for Training: %f seconds\n", (double)(end-begin) / CLOCKS_PER_SEC);
	// end = clock();
    // fprintf(stderr, "Total Elapsed for Training: %f seconds\n", (double)(end-begin) / CLOCKS_PER_SEC);
	//if (loss_file){
	//	fclose(lossfile);
	//}
	fclose(lossfile);

}


// AVE
void validate_yolt3(char *cfgfile, char *weightfile, char *valid_list_loc, 
		float iou_thresh, char *names[], char *results_dir, int nbands)
{
    int *map = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    //fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    fprintf(stderr, "test_list_loc: %s\n", valid_list_loc);
    srand(time(0));

    list *plist = get_paths(valid_list_loc);
    char **paths = (char **)list_to_array(plist);
	
    layer l = net->layers[net->n-1];
    int classes = l.classes;

    // char buff[1024];
    // char *type = option_find_str(options, "eval", "voc");
    // FILE *fp = 0;
    // FILE **fps = 0;
    // int coco = 0;
    // int imagenet = 0;
	
	// set output files
    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
		fprintf(stderr, "Output file: %s/%s.txt\n", results_dir, names[j]);		
		snprintf(buff, 1024, "%s/%s.txt", results_dir, names[j]);        
		fps[j] = fopen(buff, "w");
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    int nms = 0;
	if (iou_thresh > 0) nms=1;
    //float nms = .45; // old

    int nthreads = 1; // originally, 4
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
	args.c = nbands;
	
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    int count = 0;
    for(i = nthreads; i < m+nthreads; i += nthreads){
        //fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
			// set id as path, not the partial path
            //char *id = basecfg(path);
			char *id = path;
		    //fprintf(stderr, "path: %s\n", path);
            if (count%10 == 0){
                fprintf(stderr, "%d, test id: %s\n", i, id);
            }

            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, 0.5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, iou_thresh);
            print_yolt_detections(fps, id, dets, nboxes, classes, w, h);
            // if (coco){
            //     print_cocos(fp, path, dets, nboxes, classes, w, h);
            // } else if (imagenet){
            //     print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            // } else {
            //     print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            // }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
            count++;
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    // if(coco){
    //     fseek(fp, -2, SEEK_CUR);
    //     fprintf(fp, "\n]\n");
    //     fclose(fp);
    // }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


// AVE
void test_yolt3(char *cfgfile, char *weightfile, char *filename, float plot_thresh, float iou_thresh, 
		char *names[], image *voc_labels, int CLASSNUM, int nbands, char *results_dir)
{
	
	float hier_thresh = 0.5;
	
	//fprintf(stderr, "Label names: %s", names);
	//int i;
	//for(i = 0; i < 80; ++i){
	//	//printf("names i %i, %s\n", i, names[i]);
	//	fprintf(stderr, "%s ", names[i]);
	//}

	int show_labels_switch=1;
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(&net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    // int j;
    int nms = 0;
	if (iou_thresh > 0) nms=1;
    //float nms=.4;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    //float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    //for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image(input,0,0,nbands);
        image sized = letterbox_image(im, net->w, net->h);
		image im_three = rgb_from_multiband2(im, net->w, net->h); 
        image im_three_sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
		fprintf(stderr, "  plot thresh: %f\n", plot_thresh);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, plot_thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im_three, dets, nboxes, plot_thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        save_image_ave(im_three, results_dir, "predictions");
        free_image(im);
        free_image(sized);
		free_image(im_three);
		free_image(im_three_sized);
        if (filename) break;
    }
}
	
// AVE
void run_yolt3(int argc, char **argv)
{
	
	if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
	
	// arg 0 = GPU number
	// arg 1 'yolt'
	// arg 2 = mode 
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *test_filename = (argc > 5) ? argv[5]: 0;
	float plot_thresh = (argc > 6) ? atof(argv[6]): 0.2;
	float nms_thresh = (argc > 7) ? atof(argv[7]): 0;
    char *train_images = (argc > 8) ? argv[8]: 0;
    char *results_dir = (argc > 9) ? argv[9]: 0;
    char *valid_list_loc = (argc > 10) ? argv[10]: 0;
	char *names_str = (argc > 11) ? argv[11]: 0;
	int len_names = (argc > 12) ? atoi(argv[12]): 0;
	int nbands = (argc > 13) ? atoi(argv[13]): 0;
	char *loss_file = (argc > 14) ? argv[14]: 0;
    float min_retain_prob = (argc > 15) ? atof(argv[15]): 0.0;
	int ngpus = (argc > 16) ? atoi(argv[16]): 0;

	int reset_seen = 1;   // switch to reset the number of observations already seen

	// turn names_str into names_list
	fprintf(stderr, "\nRun YOLT3.C...\n");
	fprintf(stderr, "Plot Probablility Threshold: %f\n", plot_thresh);
	fprintf(stderr, "Label_str: %s\n", names_str);
	fprintf(stderr, "len(names): %i\n", len_names);
	fprintf(stderr, "num channels: %i\n", nbands);
	fprintf(stderr, "ngpus: %i\n", ngpus);

	// // get gpu list
    //char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    //int ngpus = 0;
    int *gpus = 0;
    int gpu = 0;
    if(ngpus > 1){
        // printf("%s\n", gpu_list);
        //int len = strlen(gpu_list);
        //ngpus = 1;
        int i;
        //for(i = 0; i < len; ++i){
        //    if (gpu_list[i] == ',') ++ngpus;
        //}
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = i;
			//gpus[i] = atoi(gpu_list);
            //gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
	//     char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	//     int *gpus = 0;
	//     int gpu = 0;
	//     int ngpus = 0;
	//     if(gpu_list){
	//         printf("%s\n", gpu_list);
	//         int len = strlen(gpu_list);
	//         ngpus = 1;
	//         int i;
	//         for(i = 0; i < len; ++i){
	//             if (gpu_list[i] == ',') ++ngpus;
	//         }
	//         gpus = calloc(ngpus, sizeof(int));
	//         for(i = 0; i < ngpus; ++i){
	//             gpus[i] = atoi(gpu_list);
	//             gpu_list = strchr(gpu_list, ',')+1;
	//         }
	//     } else {
	//         gpu = gpu_index;
	//         gpus = &gpu;
	//         ngpus = 1;
	//     }

	// turn names_str into names_list
	char **names;
	if(len_names > 0){
		names = str_split2(names_str, ',');
		fprintf(stderr, "Len names %i\n", len_names);
		//printf("names: %s", names);
		int i;
	    for(i = 0; i < len_names; ++i){
			char *ni = names[i];
			printf("label i: %i, %s\n", i, ni);
		}
		//int len_names = sizeof(*names) / sizeof(*names[0]);
		//int len_names = sizeof(names) / sizeof(*names);
		//fprintf(stderr, "Load Network:\n");
	}

	// train
    if(0==strcmp(argv[2], "train")) train_yolt3(cfg, weights, train_images, results_dir, nbands, loss_file, gpus, ngpus, reset_seen);
										//0, 1, 1);
										//gpus, ngpus, clear);						
	
	// validate
    else if(0==strcmp(argv[2], "valid")) validate_yolt3(cfg, weights, valid_list_loc, 
													   nms_thresh, names, results_dir, nbands);
	// test
	else if(0==strcmp(argv[2], "test")){
	//load  labels images
	image voc_labels[len_names];
	int i;
	for(i = 0; i < len_names; ++i){
	    char buff[256];
	    //sprintf(buff, "data/labels/%s.png", names[i]);
		//fprintf(stderr, "i, label[i] %i %s\n", i, names[i]);
	    sprintf(buff, "data/category_label_images/%s.png", names[i]);
	    voc_labels[i] = load_image_color(buff, 0, 0);
	}
	test_yolt3(cfg, weights, test_filename, plot_thresh, nms_thresh, names, voc_labels, len_names, 
			  nbands, results_dir);
	} 
													  
}
