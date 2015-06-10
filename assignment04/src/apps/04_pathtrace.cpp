#include "scene.h"
#include "intersect.h"
#include "montecarlo.h"
#include "animation.h"

#include <thread>
#include <iostream>

using namespace std;
#include <time.h>
using std::thread;

// modify the following line to disable/enable parallel execution of the pathtracer
bool parallel_pathtrace = true;

image3f pathtrace(Scene* scene, bool multithread);
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose);



// lookup texture value
vec3f lookup_scaled_texture(vec3f value, image3f* texture, vec2f uv, bool tile = true) {
    if(not texture) return value;

//    // for now, simply clamp texture coords
//        auto u = clamp(uv.x, 0.0f, 1.0f);
//        auto v = clamp(uv.y, 0.0f, 1.0f);
//        return value * texture->at(u*(texture->width()-1), v*(texture->height()-1));

    int i = (int)(texture->width() * uv.x);
    int ii = i + 1;
    int j = (int)(texture->height() * uv.y);
    int jj = j + 1;
    float s = uv.x * texture->width() - i;
    float t = uv.y * texture->height() - j;
    if(tile == true){
        i = i % texture->width();
        j = j % texture->height();
        ii = ii % texture->width();
        jj = jj % texture->height();

        i = i < 0 ? i + texture->width() : i;
        j = j < 0 ? j + texture->height() : j;
        ii = ii < 0 ? ii + texture->width() : ii;
        jj = jj < 0 ? jj + texture->height() : jj;

    }else{
        i = clamp(i, 0, texture->width()-1);
        j = clamp(j, 0, texture->height()-1);
        ii = clamp(ii, 0, texture->width()-1);
        jj = clamp(jj, 0, texture->height()-1);
    }
    auto filter = texture->at(i,j)*(1-s)*(1-t) + texture->at(i, jj)*(1-s)*t + texture->at(ii, j)*s*(1-t) + texture->at(ii, jj)*s*t;
    return filter * value;
}

// compute the brdf
vec3f eval_brdf(vec3f kd, vec3f ks, float n, vec3f v, vec3f l, vec3f norm, bool microfacet) {
    if (not microfacet) {
        auto h = normalize(v+l);
        return kd/pif + ks*(n+8)/(8*pif) * pow(max(0.0f,dot(norm,h)),n);
    } else {
//        put_your_code_here("Implement microfacet brdf");
        vec3f h = normalize(v + l);
        float d = ((n + 2) / (2 * pif)) * pow(max(0.0f, dot(norm, h)), n);
        vec3f f = ks + (one3f - ks) * pow((1 - dot(h, l)), 5);
        float g = min(min(1.0f, (2 * dot(h, norm) * dot(v, norm))/dot(v, h)), (2 * dot(h, norm) * dot(l, norm))/dot(l, h));
        vec3f p = (d * g * f)/(4 * dot(l, norm) * dot(v, norm));
        return p; // <- placeholder
    }
}

// evaluate the environment map
vec3f eval_env(vec3f ke, image3f* ke_txt, vec3f dir) {
    if(ke_txt != nullptr){
        float u = atan2(dir.x, dir.z) / (2 * pif);
        float v = 1 - acos(dir.y) / pif;
        return lookup_scaled_texture(ke, ke_txt, vec2f(u,v), true);
    }else{
        return ke;
    }
}

// compute the color corresponing to a ray by pathtrace
vec3f pathtrace_ray(Scene* scene, ray3f ray, Rng* rng, int depth) {
    // get scene intersection
    auto intersection = intersect(scene,ray);
    
    // if not hit, return background (looking up the texture by converting the ray direction to latlong around y)
    if(not intersection.hit) {
        return eval_env(scene->background, scene->background_txt, ray.d);
    }
    
    // setup variables for shorter code
    auto pos = intersection.pos;
    auto v = -ray.d;
    vec3f norm = zero3f;
    // using normal mapping
    if(intersection.mat->normal_mapping){
        auto index1 = int(intersection.texcoord.x * intersection.mat->norm_txt->width() - 1);
        auto index2 = int(intersection.texcoord.y * intersection.mat->norm_txt->height() - 1);
        norm = normalize(intersection.mat->norm_txt->at(index1, index2) * 2 - one3f);
    }else{
        norm = intersection.norm;
    }

//    // mip-map
        vec3f kd = lookup_scaled_texture(intersection.mat->kd, intersection.mat->kd_txt, intersection.texcoord);
//        message("before ismipmap");
        if (scene->isMipmap){
            image3f pic1 = read_png("yellow.png", false);
            image3f pic2 = read_png("red.png", false);
            image3f pic3 = read_png("blue.png", false);
            // calculate the distance
            float distance;
            float ratio1;
            float ratio2;
            float ratio3;
            distance = dist(intersection.pos, ray.e);
            // using the following categories to update kd
            if (distance < 1.5){
                ratio1 = 1;
                ratio2 = 0;
                ratio3 = 0;
            }
            else if (distance >= 1.5 && distance < 1.8){
                ratio2 = (distance - 1.5)/0.5;
                ratio1 = 1 - ratio2;
                ratio3 = 0;
            }
            else if (distance >= 1.8 && distance < 2){
                ratio1 = 0;
                ratio2 = 1;
                ratio3 = 0;
            }
            else if (distance >= 2 && distance < 2.5){
                ratio3 = (distance - 2)/0.5;
                ratio2 = 1-ratio3;
                ratio1 = 0;
            }
            else{
                ratio1 = 0;
                ratio2 = 0;
                ratio3 = 1;
            }
            kd = (ratio1 * lookup_scaled_texture(intersection.mat->kd, &pic1, intersection.texcoord)
                 + ratio2 * lookup_scaled_texture(intersection.mat->kd, &pic2, intersection.texcoord)
                 + ratio3 * lookup_scaled_texture(intersection.mat->kd, &pic3, intersection.texcoord)) * 5/distance;
        }
//        message("after mipmap");

    // compute material values by looking up textures
    auto ke = lookup_scaled_texture(intersection.mat->ke, intersection.mat->ke_txt, intersection.texcoord);
//         kd = lookup_scaled_texture(intersection.mat->kd, intersection.mat->kd_txt, intersection.texcoord);
    auto ks = lookup_scaled_texture(intersection.mat->ks, intersection.mat->ks_txt, intersection.texcoord);
    auto n = intersection.mat->n;
    auto mf = intersection.mat->microfacet;
    
    // accumulate color starting with ambient
    auto c = scene->ambient * kd;
    
    // add emission if on the first bounce
    if(depth == 0 and dot(v,norm) > 0) c += ke;
    
    // foreach point light
    for(auto light : scene->lights) {
        // compute light response
        auto cl = light->intensity / (lengthSqr(light->frame.o - pos));
        // compute light direction
        auto l = normalize(light->frame.o - pos);
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,l),0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
        // multiply brdf and light
        auto shade = cl * brdfcos;
        // check for shadows and accumulate if needed
        if(shade == zero3f) continue;
        // if shadows are enabled
        if(scene->path_shadows) {
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene,ray3f::make_segment(pos,light->frame.o))) c += shade;
        } else {
            // else just accumulate
            c += shade;
        }
    }
    
    // foreach surface
    for(Surface *sf: scene->surfaces){
        // skip if no emission from surface
        if(sf->mat->ke == zero3f) continue;
        // todo: pick a point on the surface, grabbing normal, area, and texcoord
        vec2f random_num2d;
        vec3f lt_position;
        vec3f lt_norm;
        float lt_area;
        vec2f lt_texcoord;

        // check if quad
        if(sf->isquad){
            // generate a 2d random number
            random_num2d = rng->next_vec2f();
            // compute light position, normal, area
            lt_position = sf->frame.o + (random_num2d.x - 0.5) * 2 *sf->radius * sf->frame.x
                          + (random_num2d.y - 0.5) * 2 *sf->radius * sf->frame.y;
            lt_norm = sf->frame.z;
            lt_area = 4 * sf->radius * sf->radius;
            // set tex coords as random value got before
            lt_texcoord = vec2f(random_num2d.x, random_num2d.y);
        }
        // else if sphere
        else{
            // generate a 2d random number
            random_num2d = rng->next_vec2f();
            // compute light position, normal, area
            vec3f dir = normalize(sample_direction_spherical_uniform(random_num2d));
            lt_position = sf->frame.o + dir * sf->radius;
            lt_norm = normalize(lt_position - sf->frame.o);
            lt_area = 4 * pif * sqr(sf->radius);
            // set tex coords as random value got before
            lt_texcoord = vec2f(random_num2d.x, random_num2d.y);
        }
        // get light emission from material and texture
        vec3f ke = lookup_scaled_texture(sf->mat->ke, sf->mat->ke_txt, lt_texcoord);
        // compute light direction
        vec3f lt_direction = normalize(lt_position - pos);
        // compute light response (ke * area * cos_of_light / dist^2)
        vec3f lt_response = ke * lt_area * max(0.0f, -1 * dot(lt_norm, lt_direction)) / distSqr(lt_position, pos);
        // compute the material response (brdf*cos)
        vec3f material_response = max(0.0f, dot(lt_direction, norm)) * eval_brdf(kd,ks,n,v,lt_direction, norm, mf);
        // multiply brdf and light
        vec3f res = lt_response * material_response;
        // check for shadows and accumulate if needed
        if(res == zero3f){
            continue;
        }
        // if shadows are enabled
        if(scene->path_shadows){
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene, ray3f::make_segment(pos, lt_position))){
                c += res;
            }
        }else{// else just accumulate
           c += res;
        }
    }
    
    // todo: sample the brdf for environment illumination if the environment is there
    // if scene->background is not zero3f
    if(scene->background != zero3f){
        // pick direction and pdf
        vec2f random_num2d = rng->next_vec2f();
        auto temp = sample_brdf(kd, ks, n, v, norm, random_num2d, rng->next_float());
        vec3f dir = temp.first;
        float pdf = temp.second;
        // compute the material response (brdf*cos)
        vec3f material_response = max(0.0f, dot(dir, norm)) * eval_brdf(kd, ks, n, v, dir, norm, mf);
        // todo: accumulate response scaled by brdf*cos/pdf
        vec3f res = material_response * eval_env(scene->background, scene->background_txt, dir) / pdf;
        // if material response not zero3f and if shadows are enabled
        if(res != zero3f && scene->path_shadows){
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene, ray3f(pos, dir))){
                c += res;
            }
        // else just accumulate
        }else{
            c += res;
        }
    }
    // todo: sample the brdf for indirect illumination
    // pick direction and pdf
    vec2f random_num2d = rng->next_vec2f();
    vec3f dir = sample_brdf(kd, ks, n, v, norm, random_num2d, rng->next_float()).first;
    float brdf = sample_brdf(kd, ks, n, v, norm, random_num2d, rng->next_float()).second;
    // compute the material response (brdf*cos)
    vec3f material_response = max(0.0f, dot(dir, norm)) * eval_brdf(kd, ks, n, v, dir, norm, mf);

    // if use Russian
    if(scene->isRussian && brdf > 0.1){
        c += pathtrace_ray(scene, ray3f(pos, dir), rng, 1 + depth) * material_response / (1 - brdf);
    }else{
        // if kd and ks are not zero3f and haven't reach max_depth
        if((kd != zero3f || ks != zero3f) && depth < scene->path_max_depth){
            // accumulate recersively scaled by brdf*cos/pdf
//            auto tmp = material_response * pathtrace_ray(scene, ray3f(pos, dir), rng, 1 + depth) / brdf;
//            cout<<"t=("<<tmp.x<<","<<tmp.y<<","<<tmp.z<<endl;
            c += material_response * pathtrace_ray(scene, ray3f(pos, dir), rng, 1 + depth) / brdf;
        }
    }
    // if the material has reflections
    if(not (intersection.mat->kr == zero3f)) {
        // if not blurry
        if(!scene->isBlurry){
            // create the reflection ray
            auto rr = ray3f(intersection.pos,reflect(ray.d,intersection.norm));
            // accumulate the reflected light (recursive call) scaled by the material reflection
            c += intersection.mat->kr * pathtrace_ray(scene,rr,rng,depth+1);
        }else{
            // using samp 10
            vec3f reflection_dir;
            vec2f random_num2d;
            for(int i = 0; i < 10 ; i++){
                reflection_dir = reflect(ray.d, intersection.norm);
                random_num2d = rng->next_vec2f();
                //using the formula from the slide
                //get u,v
                vec3f u = normalize(cross(ray.d, reflection_dir));
                vec3f v = normalize(cross(reflection_dir, u));

                //using blur 0.05
                reflection_dir = normalize(reflection_dir + (0.5 - random_num2d.x) * 0.05 * u
                                                          + (0.5 - random_num2d.y) * 0.05 * v);
                auto rr = ray3f(intersection.pos,reflection_dir);
                // accumulate the reflected light (recursive call) scaled by the material reflection
                c += intersection.mat->kr * pathtrace_ray(scene,rr,rng,depth+1) / 10;
            }
        }
    }
    
    // return the accumulated color
    return c;
}


// runs the raytrace over all tests and saves the corresponding images
int main(int argc, char** argv) {
    auto args = parse_cmdline(argc, argv,
        { "04_pathtrace", "raytrace a scene",
            {  {"resolution",     "r", "image resolution", typeid(int),    true,  jsonvalue() } },
            {  {"scene_filename", "",  "scene filename",   typeid(string), false, jsonvalue("scene.json") },
               {"image_filename", "",  "image filename",   typeid(string), true,  jsonvalue("") } }
        });
    
    auto scene_filename = args.object_element("scene_filename").as_string();
    Scene* scene = nullptr;
    if(scene_filename.length() > 9 and scene_filename.substr(0,9) == "testscene") {
        int scene_type = atoi(scene_filename.substr(9).c_str());
        scene = create_test_scene(scene_type);
        scene_filename = scene_filename + ".json";
    } else {
        scene = load_json_scene(scene_filename);
    }
    error_if_not(scene, "scene is nullptr");
    
    auto image_filename = (args.object_element("image_filename").as_string() != "") ?
        args.object_element("image_filename").as_string() :
        scene_filename.substr(0,scene_filename.size()-5)+".png";
    
    if(not args.object_element("resolution").is_null()) {
        scene->image_height = args.object_element("resolution").as_int();
        scene->image_width = scene->camera->width * scene->image_height / scene->camera->height;
    }
    // the start of time
    double start_time = clock();
    
    // NOTE: acceleration structure does not support animations
    message("reseting animation...\n");
    animate_reset(scene);
    
    message("accelerating...\n");
    accelerate(scene);
    
    message("rendering %s...\n", scene_filename.c_str());
    auto image = pathtrace(scene, parallel_pathtrace);
    
    double end_time = clock();
    //compute the time
    double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\nconsumption duration: %f\n", duration);

    message("saving %s...\n", image_filename.c_str());
    write_png(image_filename, image, true);
    
    delete scene;
    message("done\n");
}


/////////////////////////////////////////////////////////////////////
// Rendering Code


// pathtrace an image
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose) {
    if(verbose) message("\n  rendering started        ");
    // foreach pixel
    for(auto j = offset_row; j < scene->image_height; j += skip_row ) {
        if(verbose) message("\r  rendering %03d/%03d        ", j, scene->image_height);
        for(auto i = 0; i < scene->image_width; i ++) {
            // init accumulated color
            image->at(i,j) = zero3f;
            // grab proper random number generator
            auto rng = &rngs->at(i, j);
            // foreach sample
            for(auto jj : range(scene->image_samples)) {
                for(auto ii : range(scene->image_samples)) {
                    // compute ray-camera parameters (u,v) for the pixel and the sample
                    auto u = (i + (ii + rng->next_float())/scene->image_samples) /
                        scene->image_width;
                    auto v = (j + (jj + rng->next_float())/scene->image_samples) /
                        scene->image_height;
                    if(scene->depth_of_field){
                        int depth = 8;
                        vec3f color = zero3f;
                        for(int i = 0; i < depth; i++){
                            vec2f random_num2d = rng->next_vec2f();
                            vec3f F = 0.1f * vec3f(random_num2d.x, random_num2d.y, 0);
                            float lp = scene->camera->focus / scene->camera->dist;
                            vec3f Q = vec3f((u-0.5f)*scene->camera->width * lp, (v-0.5f)*scene->camera->height * lp, -scene->camera->focus);
                            ray3f ray = transform_ray(scene->camera->frame, ray3f(F, normalize(Q - F)));
                            color += pathtrace_ray(scene,ray,rng,0);
                        }
                        // set pixel to the color raytraced with the ray
                        image->at(i,j) += color/depth;
                    }else{
                        // compute camera ray
                        auto ray = transform_ray(scene->camera->frame,
                            ray3f(zero3f,normalize(vec3f((u-0.5f)*scene->camera->width,
                                                         (v-0.5f)*scene->camera->height,-1))));
                        // set pixel to the color raytraced with the ray
                        image->at(i,j) += pathtrace_ray(scene,ray,rng,0);
                    }
                }
            }
            // scale by the number of samples
            image->at(i,j) /= (scene->image_samples*scene->image_samples);
        }
    }
    if(verbose) message("\r  rendering done        \n");
    
}

// pathtrace an image with multithreading if necessary
image3f pathtrace(Scene* scene, bool multithread) {
    // allocate an image of the proper size
    auto image = image3f(scene->image_width, scene->image_height);
    
    // create a random number generator for each pixel
    auto rngs = RngImage(scene->image_width, scene->image_height);

    // if multitreaded
    if(multithread) {
        // get pointers
        auto image_ptr = &image;
        auto rngs_ptr = &rngs;
        // allocate threads and pathtrace in blocks
        auto threads = vector<thread>();
        auto nthreads = thread::hardware_concurrency();
        for(auto tid : range(nthreads)) threads.push_back(thread([=](){
            return pathtrace(scene,image_ptr,rngs_ptr,tid,nthreads,tid==0);}));
        for(auto& thread : threads) thread.join();
    } else {
        // pathtrace all rows
        pathtrace(scene, &image, &rngs, 0, 1, true);
    }
    
    // done
    return image;
}


