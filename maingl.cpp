/**
 *
 *   To convert exr to png.
 *   ./maingl --exr
 *
 *   To render uv map from "objuv"
 *   ./maingl --autogen
 *
 *   To render a textured mesh from "obj"
 *   ./maingl --textured --autogen
 *
 * */
#include <cmath>
#include <sys/time.h>
#include <math.h>
#include <cstdio>
#include <iostream>
#include <queue>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "gflags/gflags.h"
#include "viz.h"
#include "json11.hpp"

using namespace std;
using namespace cv;
using namespace json11;
//using namespace std;

//int W = 2256, H = 1504;

int w, h;
float f, px, py;
int mvid = 0;
float ccx = 0,ccy = 0.1,ccz=0;
float dx = 0.01,dy = 0.01,dz=0.001;
float cc = 0;
struct Camera {
  Mat r, t;
  string poseId;
  string path;
  Camera(Mat &r, Mat &t, string poseId) : r(r), t(t), poseId(poseId) { }
  Camera(Mat &r, Mat &t, string poseId, string path) : r(r), t(t), poseId(poseId), path(path) { }
};
struct PointRGB {
  float x, y, z;
  int r, g, b;
  PointRGB(float x, float y, float z, int r, int g, int b) : x(x), y(y), z(z), r(r), g(g), b(b) {}
};


float dmin = 5, drange = 1;

Mat planes[4];

int fixview = 0;
vector<Camera> cams;
vector<PointRGB> points;
string outputFolder;

Mat ki, mvi;

GLuint fbo, render_buf, depth_buf;
int mode=0;


DEFINE_string(dataset, "greentea", "camera file");
DEFINE_string(ref, "0016", "camera file");


float delta = 0.1;
void keyboard(unsigned char key, int x, int y) {
  printf("%d\n", key);
  if (key == 'a') {
    if(mode == 0){ccx-=delta;}
    else if(mode == 1){ccy-=delta;}
    else if(mode == 2){ccz-=delta;}

  } else if (key == 'd') {
    if(mode == 0){ccx+=delta;}
    else if(mode == 1){ccy+=delta;}
    else if(mode == 2){ccz+=delta;}

  } else if (key == 'w') {
    if(mode == 0){dx+=delta;}
    else if(mode == 1){dy+=delta;}
    else if(mode == 2){dz+=delta;}
  } else if (key == 's') {
    if(mode == 0){dx-=delta;}
    else if(mode == 1){dy-=delta;}
    else if(mode == 2){dz-=delta;}
  } else if (key == 'j') {
    delta *= 0.5;
  } else if (key == 'u') {
    delta *= 2;
  } else if (key == 'x') {
    fixview ^= 1;
  } else if (key == 'o') {
    if (mode == 0) {
      FILE *fo = fopen(("../datasets/" + FLAGS_dataset + "/planes.txt").c_str(), "w");
      fprintf(fo, "%f %f\n", ccx, dx);
      fclose(fo);
      printf("file written %f %f\n", ccx,  dx);
      mode = 1;
    } else if (mode == 1) {
      FILE *fo = fopen(("../datasets/" + FLAGS_dataset + "/planes.txt").c_str(), "a");
      fprintf(fo, "%f %f\n", ccy, dy);
      fclose(fo);
      printf("file written %f %f\n", ccy,  dy);
      mode = 2;
    } else if (mode == 2){
      FILE *fo = fopen(("../datasets/" + FLAGS_dataset + "/planes.txt").c_str(), "a");
      fprintf(fo, "%f %f\n", ccz, dz);
      fclose(fo);
      printf("file written %f %f\n", ccz,  dz);
      mode = 0;
    }

  }
  printf("dmin = %f\ndmax = %f\ndelta = %f\n", dmin, dmin+drange, delta);
  Viz::display();

}

void setP() {
  Mat p = (Mat_<float>(4, 4) <<
      f * 2 / w, 0           , px * 2 / w - 1   , 0 ,
      0        , -(f * 2 / h), -(py * 2 / h - 1), 0 ,
      0        , 0           , 1                , -1,
      0        , 0           , 1                , 0);

  p = p.t();
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity();
  glMultMatrixf((float*)p.data);
}

void setMV(int id) {
  printf("setmv %d\n", id);
  Mat rot = cams[id].r;
  Mat t = cams[id].t;

  Mat mv = Mat::zeros(4, 4, CV_32F);
  rot = rot.t();
  t = - rot * t;

  rot.copyTo(mv(Range(0, 3), Range(0, 3)));
  t.copyTo(mv(Range(0, 3), Range(3, 4)));
  mv.at<float>(3, 3) = 1;
  mv = mv.t();

  cout << mv << endl;

  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity();
  glMultMatrixf((float*)mv.data);
}

void modelview() {
  if (fixview) {
    setMV(mvid);
    setP();
  }
}

bool sortByPath(const Camera & lhs, const Camera & rhs ) {
   return lhs.path.compare(rhs.path) < 0;
}

void readJSON(string jsonfile) {
  std::ifstream t(jsonfile);
  std::string str((std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
  string err;
  Json json = Json::parse(str, err);
  auto v = json["poses"];
  for (auto &a : v.array_items()) {
    Mat r(3, 3, CV_32F);
    Mat t(3, 1, CV_32F);
    const auto rv = a["pose"]["transform"]["rotation"];
    forMat(i, j, r)
      r.at<float>(i, j) = stof(rv[i*3+j].string_value());
    const auto tv = a["pose"]["transform"]["center"];
    forMat(i, j, t)
      t.at<float>(i, j) = stof(tv[i].string_value());

    cams.push_back(Camera(r, t, a["poseId"].string_value()));
    cout << "poseId: " << a["poseId"].dump() << endl;
    cout << "path: " << a["path"].dump() << endl;
    cout << r << endl;
    cout << t << endl;
  }

  for (auto &a : json["views"].array_items()) {
    string id = a["poseId"].string_value();
    for (int i = 0; i < cams.size(); i++) {
      if (cams[i].poseId == id) {
        cams[i].path = a["path"].string_value();
        break;
      }
    }
  }

  w = stoi(json["intrinsics"][0]["width"].string_value());
  h = stoi(json["intrinsics"][0]["height"].string_value());
  f = stof(json["intrinsics"][0]["pxFocalLength"].string_value());
  px = stof(json["intrinsics"][0]["principalPoint"][0].string_value());
  py = stof(json["intrinsics"][0]["principalPoint"][1].string_value());
  printf("w,h = %d %d\nf = %f\npx,py = %f %f\n", w, h, f, px, py);

  sort(cams.begin(), cams.end(), sortByPath);
  for (int i = 0; i < cams.size(); i++) {
    printf("%s %s\n", cams[i].path.c_str(), cams[i].poseId.c_str());
  }
}

string runPython(string cmd) {
  char val[512];
  FILE *fp = popen(cmd.c_str() , "r");
  fscanf(fp, "%s", val);
  pclose(fp);
  return val;
}

void readPly(string file) {
  FILE *f;
  char st[256];

  f = fopen (file.c_str() , "r");
  int start = 0;
  while ( fgets (st , 256 , f) != NULL ) {
    //printf("%s\n", mystring);
    if (start) {
      float x, y, z;
      int r, g, b;
      sscanf(st, "%f %f %f %d %d %d", &x, &y, &z, &r, &g, &b);
      //printf("%f %f %f %d %d %d\n", x, y, z, r, g, b);
      points.push_back(PointRGB(x, y, z, r, g, b));
      if (g<254){
        ccx += x;
        ccy += y;
        ccz += z;
        dx += x*x;
        dy += y*y;
        dz += z*z;
        cc += 1;
      }
    }
    if (strcmp(st, "end_header\n") == 0) {
      start = 1;
    }
  }
  fclose (f);
  ccx /=cc;
  ccy /=cc;
  ccz /=cc;
  dx /=cc;
  dy /=cc;
  dz /=cc;
  dx = sqrt(dx-ccx*ccx)*2;
  dy = sqrt(dy-ccy*ccy)*2;
  dz = sqrt(dz-ccz*ccz)*2;
  cout << ccx<<" "<<ccy<<" "<<ccz<<endl;
  cout << dx<<" "<<dy<<" "<<dz<<endl;
}


void findMedianCenter() {

  vector<float> cx, cy, cz;
  for (PointRGB &p : points) {
    if (!(p.r == 0 && p.g == 255 && p.b == 0)) {
      cx.push_back(p.x);
      cy.push_back(p.y);
      cz.push_back(p.z);
    }
  }
  sort(cx.begin(), cx.end());
  sort(cy.begin(), cy.end());
  sort(cz.begin(), cz.end());
  Viz::ox = cx[cx.size() / 2];
  Viz::oy = cy[cy.size() / 2];
  Viz::oz = cz[cz.size() / 2];
}

void setInverseKM(int id) {
  Mat rot = cams[id].r;
  Mat t = cams[id].t;

  Mat mv = Mat::zeros(4, 4, CV_32F);
  rot = rot.t();
  t = - rot * t;

  rot.copyTo(mv(Range(0, 3), Range(0, 3)));
  t.copyTo(mv(Range(0, 3), Range(3, 4)));
  mv.at<float>(3, 3) = 1;


  Mat k = (Mat_<float>(4, 4) <<
      f * 2 / w, 0           , px * 2 / w - 1   , 0 ,
      0        , -(f * 2 / h), -(py * 2 / h - 1), 0 ,
      0        , 0           , 1                , 0,
      0        , 0           , 0                , 1);

  mvi = mv.inv();
  ki = k.inv();
}

int findIdOfPath(string name) {
  for (int i = 0; i < cams.size(); i++) {
    Camera cam = cams[i];
    if (cam.path.find(name) != std::string::npos) {
      return i;
    }
  }
  printf("not found\n");
  exit(0);
  return -1;
}

void initialize() {
  string camera = runPython("python3 -c 'import glob; print(glob.glob(\"../datasets/" + FLAGS_dataset + "/*/Stru*/*/cameras.sfm\")[0])'");
  string ply = runPython("python3 -c 'import glob; print(glob.glob(\"../datasets/" + FLAGS_dataset + "/*/Stru*/*/*.ply\")[0])'");

  readJSON(camera);
  readPly(ply);

  findMedianCenter();
  setInverseKM(findIdOfPath(FLAGS_ref));

  FILE *fi = fopen(("../datasets/" + FLAGS_dataset + "/planes.txt").c_str(), "r");
  if (fi) {
    float tmp;
    fscanf(fi, "%f %f", &dmin, &tmp);
    drange = tmp - dmin;
    fclose(fi);
  }

}

void display() {
  glPointSize(4);
  for (PointRGB &p : points) {
    glBegin(GL_POINTS);
    glColor3ub(p.r, p.g, p.b);
    glVertex3f(p.x, p.y, p.z);
    glEnd();
  }

  int v[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  if (mode == 0){
    glColor4ub(255, 255, 0, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx+dx, ccy+ dy*v[i][0],  ccz+ dz*v[i][1]);
    } glEnd();
    glColor4ub(255, 0, 0, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx-dx, ccy+ dy*v[i][0],  ccz+ dz*v[i][1]);
    } glEnd();

  } else if (mode == 1){
    glColor4ub(0, 0, 255, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx+dx*v[i][0], ccy+ dy,  ccz+ dz*v[i][1]);
    }
    glEnd();

    glColor4ub(0, 255, 255, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx+dx*v[i][0], ccy - dy,  ccz+ dz*v[i][1]);
    }
    glEnd();

  }else if (mode == 2){
    glColor4ub(255, 255, 0, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx+dx*v[i][1], ccy+ dy*v[i][0],  ccz+ dz);
    } glEnd();
    glColor4ub(255, 0, 0, 128);
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i < 4; i++)  {
      glVertex3f(ccx+dx*v[i][1], ccy+ dy*v[i][0],  ccz- dz);
    } glEnd();

  }


}


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  initialize();
  printf("Hu Hu!");
  Viz::setModelviewCallback(modelview);
  //Viz::normalizeSurface();
  //Viz::useTexture = 0;
  //Viz::useColor = 1;
  Viz::setKeyboardCallback(keyboard);
  Viz::setDisplayCallback(display);


  Viz::startWindow(w * 0.5, h * 0.5);
}
