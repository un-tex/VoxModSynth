// --------------------------------------------------------------
// Implementation in C++ of 3D model synthesis / wave function collapse
// Simple, quite efficient and easily hackable
//
// For detailed usage information see README.md
// - the expected voxel format is '.slab.vox' as exported by MagicaVoxel
// - the output can be directly imported into MagicaVoxel
//   (use MagicaVoxel viewer for larger outputs, as MagicaVoxel clamps to 128^3)
// - palette indices are used as tile ids (labels).
// - palette index 255 is empty, 254 is ground.
// - input files are in subdir exemplars/
// - output is produced in subdir results/
//    results/synthesized.slab.vox is the synthesized labeling
//    results/synthesized_detailed.slab.vox is the output using detailed tiles
//
// For more details on model synthesis:
// - http://graphics.stanford.edu/~pmerrell/
// - https://github.com/mxgmn/WaveFunctionCollapse
// 
// The goal is to keep it short, efficient, and (relatively) clear.
// Shamelessly uses globals.
//
// Enjoy!
//
// Limitations:
// - all labels are currently equiprobable (will be updated soon)
//
// Sylvain Lefebvre @sylefeb
// --------------------------------------------------------------
/*
MIT License
https://opensource.org/licenses/MIT

Copyright 2017, Sylvain Lefebvre

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files(the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sub-license, and / or sell 
copies of the Software, and to permit persons to whom the Software is furnished 
to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
*/
// --------------------------------------------------------------

#include <LibSL/LibSL.h>

#include <iostream>
#include <ctime>
#include <cmath>
#include <set>
#include <map>
#include <queue>
#include <limits>
#include <cstring>

#define ID_EMPTY    255
#define ID_GROUND   254
#define ID_OBJECT   253
#define ID_CONTACT  252

// --------------------------------------------------------------

using namespace std;

// --------------------------------------------------------------

// volume size to synthesize (sz^3)
const int   sz = 32;

bool use_scene = true;

// name of the problem (files in subdirectory exemplars/)
string problem = use_scene ? "test-scene" : "test";

// name of the initial scene (object + contact points)
string scene = "scene-32";

// synthesize a periodic structure? (only makes sense if not using borders!)
const bool  periodic = false;

// --------------------------------------------------------------

// number of labels in problem
int            num_lbls;

// bits to describe axial directions
const uchar    axis_x = 1;
const uchar    axis_y = 2;
const uchar    axis_z = 4;

// for navigating neighbors
const v3i      neighs[6] = { v3i(-1, 0, 0), v3i(1, 0, 0), v3i(0, -1, 0), v3i(0, 1, 0), v3i(0, 0, -1), v3i(0, 0, 1) };
const bool     side[6] = { false, true, false, true, false, true };
const uchar    face[6] = { axis_x, axis_x, axis_y, axis_y, axis_z, axis_z };
const int      n_left = 0;
const int      n_right = 1;
const int      n_back = 2;
const int      n_front = 3;
const int      n_below = 4;
const int      n_above = 5;

string dirToStr(int dir) {
  switch (dir) {
  case n_left:
    return "left";
  case n_right:
    return "right";
  case n_back:
    return "back";
  case n_front:
    return "front";
  case n_below:
    return "below";
  case n_above:
    return "above";  
  default:
    break;
  }

  return "";
}

// constraint bit-field for each label pairs
// (e.g. constraints.at(2,5) = axis_x means label 2 can have 5 on its right)
Array2D<uchar> constraints;
// same as above under a different form allowing for faster checks
// (build from 'constraints' by prepareFastConstraintChecks)
Array< Array<vector<int> > > allowed_by_side;

// information from loaded voxel problem
Array<v3b>               palette; // RGB palette
map<uchar, int>          pal2id;  // palette index to label id
map<int, uchar>          id2pal;  // label id to palette index

/* -------------------------------------------------------- */

// This prepares the small data structure 'allowed_by_side' from 'constraints'
// to allow for a faster check in 'updateConstraintsAtSite'
void prepareFastConstraintChecks()
{
  allowed_by_side.allocate(6);
  ForIndex(n, 6) {
    allowed_by_side[n].allocate(num_lbls);
    ForIndex(l1, num_lbls) {
      bool allowed = false;
      ForIndex(l2, num_lbls) {
        int a = l1; int b = l2;
        if (!side[n]) { std::swap(a, b); }
        bool can_be_side_by_side = (constraints.at(a, b) & face[n]);
        if (can_be_side_by_side) {
          allowed_by_side[n][l1].push_back(l2);
        }
      }
    }
  }
}

// --------------------------------------------------------------

// Returns the opposite neighbor, used in 'updateConstraintsAtSite'
inline int oppositeNeighbor(int n)
{
  switch (n)
  {
  case n_left:  return n_right; break;
  case n_right: return n_left;  break;
  case n_back:  return n_front; break;
  case n_front: return n_back;  break;
  case n_below: return n_above; break;
  case n_above: return n_below; break;
  }
  return -1;
}

// --------------------------------------------------------------

// Tiny class to hold a vector of bools representing choices at a site (voxel)
// encoded as a bit field (an unsigned int holds 32 bits for the first 32 labels)

class Presence
{
private:
  static const int c_MaxLabelFields = 2;        // enough to hold 64 labels, increase if needed
  static const int s_PowNumBits = 5;            // sizeof(uint) * 8 = 32 = 2^5
  static const int s_ModNumBits = (1 << 5) - 1; // 31
  uint m_Values[c_MaxLabelFields];
public:
  Presence() { }
  Presence&  operator = (const Presence& p) { memcpy(m_Values, p.m_Values, c_MaxLabelFields * sizeof(uint)); return *this; }
  const bool operator[](int n) const   { return (m_Values[n >> s_PowNumBits] >> (n & s_ModNumBits)) & 1; }
  void       set(int n, bool b) { 
    if (b) { m_Values[n >> s_PowNumBits] |=   1 << (n & s_ModNumBits); } 
    else   { m_Values[n >> s_PowNumBits] &= ~(1 << (n & s_ModNumBits)); } 
  }
  void fill(bool b) { memset(m_Values, b ? 0xFF : 0x00, c_MaxLabelFields * sizeof(uint)); }
  friend ostream& operator<<(ostream& os, const Presence& p);
};

ostream& operator<<(ostream& os, const Presence& p) {
  string sep = ", ";
  ForIndex(lbl, num_lbls) {
    if (p[lbl]) std::cout << id2pal[lbl] + 1 << sep;
  }
  return os;
}

// --------------------------------------------------------------
// Simple helper functions to manipulate Presence vectors
//
// Note the use of the global 'num_lbls'. Yes, not elegant, but avoids 
// storing the size (which remains the same after loading a given problem)
// in the Presence class.

inline bool isFalse(const Presence& a)
{
  ForIndex(i, num_lbls) { if (a[i]) return false; }
  return true;
}

inline void orEq(Presence& a, const Presence& b)
{
  ForIndex(i, num_lbls) { a.set(i, a[i] || b[i]); }
}

inline void andEq(Presence& a, const Presence& b)
{
  ForIndex(i, num_lbls) { a.set(i, a[i] && b[i]); }
}

/* -------------------------------------------------------- */

// Updates the set of possible labels at a given site (voxel i,j,k), considering the n-th neighbor.
// Returns whether something changed, and whether all labels disappeared due to over-constraints (failed).
// This is a local update used in the global 'propagateConstraints' function below.
void updateConstraintsAtSite(int i, int j, int k, int n, Array3D<Presence>& _S, bool& _changed, bool& _failed)
{
  if (!periodic) {
    if ( (i + neighs[n][0] < 0 || i + neighs[n][0] >= (int)_S.xsize())
      || (j + neighs[n][1] < 0 || j + neighs[n][1] >= (int)_S.ysize())
      || (k + neighs[n][2] < 0 || k + neighs[n][2] >= (int)_S.zsize())) {
      // out of domain, nothing changes
      _changed = false;
      _failed = false;
      return;
    }
  }

  _changed = false;
  const Presence& from_neigh = _S.at<Wrap>(i + neighs[n][0], j + neighs[n][1], k + neighs[n][2]);
  ForIndex(l1, num_lbls) {
    if (_S.at(i, j, k)[l1]) {
      bool allowed = false;
      int num = (int)allowed_by_side[n][l1].size();
      ForIndex(t, num) {
        int l2 = allowed_by_side[n][l1][t];
        allowed = allowed || from_neigh[l2];
      }
      if (!allowed) {
        _changed = true;
        _S.at(i, j, k).set(l1,false);
      }
    }
  }

  // is the selection empty?
  if (isFalse(_S.at(i,j,k))) { // yes ...
    _failed = true;
    //std::cout << "Failed:\t" << i << "\t" << j << "\t" << k << "\n"; 
  } else {
    _failed = false;
  }  
}

/* -------------------------------------------------------- */

// Propagates the constraints: this is the major ingredient of model synthesis.
// Initially all labels are present (possible). When some labels are discarded,
// some choices are no longer possible in the neighbors due to the constraints. 
// This function will propagate the change throughout the entire domain.
bool propagateConstraints(int i, int j, int k, Array3D<Presence>& _S)
{
  std::queue<v3i> q;
  q.push(v3i(i, j, k));
  while (!q.empty()) {
    v3i cur = q.front();
    q.pop();
    // update neighbors
    ForIndex(n, 6) {
      v3i ne = v3i(cur[0] + neighs[n][0], cur[1] + neighs[n][1], cur[2] + neighs[n][2]);
      ne[0] = (ne[0] + _S.xsize()) % _S.xsize();
      ne[1] = (ne[1] + _S.ysize()) % _S.ysize();
      ne[2] = (ne[2] + _S.zsize()) % _S.zsize();
      bool changed;
      bool failed;
      updateConstraintsAtSite(ne[0], ne[1], ne[2], oppositeNeighbor(n), _S, changed, failed);
      if (changed) {
        q.push(ne); // changed: add to sites to process
      }
      if (failed) {
        std::cout << "Failed: (" << i << ",\t" << j << ",\t" << k << ",\t" << n << ")\n";
        return false; // constraints disagree, fail
      }
    }
  }
  return true;
}

/* -------------------------------------------------------- */

// Initializes the domain with a 'soup' where all labels are possible.
// If lbl_empty is given, an empty border is initialized all around the domain.
bool init_global_soup(Array3D<Presence>& S,int lbl_empty = -1)
{
  // init: global, uniform soup
  ForArray3D(S, i, j, k) {
    S.at(i, j, k).fill(true);
  }
  bool ok = true;
  if (lbl_empty > -1) {
    // border
    ForIndex(k, S.zsize()) {
      ForIndex(i, S.xsize()) {
        S.at(i, 0, k).fill(false);
        S.at(i, 0, k).set(lbl_empty,true);
        S.at(i, S.ysize() - 1, k).fill(false);
        S.at(i, S.ysize() - 1, k).set(lbl_empty,true);
        ok &= propagateConstraints(i, 0, k, S);
        ok &= propagateConstraints(i, S.ysize() - 1, k, S);
      }
      ForIndex(j, S.ysize()) {
        S.at(0, j, k).fill(false);
        S.at(0, j, k).set(lbl_empty,true);
        S.at(S.xsize() - 1, j, k).fill(false);
        S.at(S.xsize() - 1, j, k).set(lbl_empty,true);
        ok &= propagateConstraints(0, j, k, S);
        ok &= propagateConstraints(S.xsize() - 1, j, k, S);
      }
    }
    ForIndex(j, S.xsize()) {
      ForIndex(i, S.xsize()) {
        S.at(i, j, 0).fill(false);
        S.at(i, j, 0).set(lbl_empty,true);
        S.at(i, j, S.zsize() - 1).fill(false);
        S.at(i, j, S.zsize() - 1).set(lbl_empty,true);
        ok &= propagateConstraints(i, j, 0, S);
        ok &= propagateConstraints(i, j, S.zsize() - 1, S);
      }
    }
  }
  return ok; // could fail due to propagation
}

/* -------------------------------------------------------- */

// Initializes the domain with an empty assignment.
// If lbl_ground is given, a ground is created on z == 0
bool init_global_empty(Array3D<Presence>& S, int lbl_empty,int lbl_ground=-1)
{
  if (lbl_ground < 0) lbl_ground = lbl_empty;
  ForArray3D(S, i, j, k) {
    S.at(i, j, k).fill(false);
    if (k > 0) {
      S.at(i, j, k).set(lbl_empty,true);
    } else {
      S.at(i, j, k).set(lbl_ground,true);
    }
  }
  return true;
}

bool init_only_scene(Array3D<Presence>& S, Array3D<uchar>& scene) {
  // ForArray3D(S, i, j, k) {
  //   S.at(i, j, k).fill(false);
  //   if ((int) scene.at(i, j, k) == ID_OBJECT) { // object
  //     S.at(i, j, k).set(pal2id[ID_OBJECT], true);
  //   } else if ((int) scene.at(i, j, k) == ID_CONTACT) { // contact
  //     S.at(i, j, k).set(pal2id[ID_CONTACT], true);
  //   } else {
  //     S.at(i, j, k).set(pal2id[ID_EMPTY], true);
  //   }
  // }

  ForArray3D(S, i, j, k) {
    // int rev_k = sz - 1 - k;
    S.at(i, j, k).fill(false);
    int pal = (int) scene.at(i, j, k);
    S.at(i, j, k).set(pal2id[pal], true);
  }

  return true;
}

/* -------------------------------------------------------- */
// Start from soup, add scene constraints, propagate
bool init_global_scene(Array3D<Presence>& S, Array3D<uchar>& scene) {
  bool ok = true;

  ForArray3D(S, i, j, k) {
    S.at(i, j, k).fill(true); // soup
    // S.at(i, j, k).set(pal2id[ID_OBJECT], false);
    // S.at(i, j, k).set(pal2id[ID_CONTACT], false);
    // S.at(i, j, k).set(pal2id[ID_GROUND], false);
  }

  ForArray3D(S, i, j, k) {
    switch((int) scene.at(i, j, k)) {
      case ID_OBJECT:
        S.at(i, j, k).fill(false);
        S.at(i, j, k).set(pal2id[ID_OBJECT], true);
        ok &= propagateConstraints(i, j, k, S);
        break;
      case ID_CONTACT:
        S.at(i, j, k).fill(false);
        S.at(i, j, k).set(pal2id[ID_CONTACT], true);
        ok &= propagateConstraints(i, j, k, S);
        break;
      case ID_GROUND:
        S.at(i, j, k).fill(false);
        S.at(i, j, k).set(pal2id[ID_GROUND], true);
        ok &= propagateConstraints(i, j, k, S);
        break;
      default:
        if (i == 0 || i == sz - 1 ||
            j == 0 || j == sz - 1) {
          S.at(i, j, k).fill(false);
          S.at(i, j, k).set(pal2id[ID_EMPTY], true);
          //ok &= propagateConstraints(i, j, k, S);
        } else {
          S.at(i, j, k).set(pal2id[ID_OBJECT], false);
          S.at(i, j, k).set(pal2id[ID_CONTACT], false);
          S.at(i, j, k).set(pal2id[ID_GROUND], false);
        }
        break;
    }
  }

  // ForArray3D(S, i, j, k) {
  //   ok &= propagateConstraints(i, j, k, S);
  // }

  //ok &= propagateConstraints(0, 0, 0, S);

  // ok &= propagateConstraints(13, 0, 14, S);
  // ok &= propagateConstraints(2, 3, 14, S);
  // ok &= propagateConstraints(9, 6, 14, S);
  // ok &= propagateConstraints(5, 9, 14, S);
  // ok &= propagateConstraints(15, 9, 14, S);
  // ok &= propagateConstraints(10, 11, 14, S);

  ForArray3D(S, i, j, k) {
    std::cout << "(" << i << ", " << j << ", " << k << "):\t" << S.at(i, j, k) << "\n";
  }

  return ok;
}

/* -------------------------------------------------------- */

// Resets a sub-domain with an empty soup. The border is preserved and
// constraints are propagated inside.
// Returns true on success, false otherwise (i.e. constraints cannot be resolved).
// The domain is changed, even on failure. Caller is responsible for restoring it.
bool reinit_sub(Array3D<Presence>& S, int lbl_empty, AAB<3, int> sub)
{
  // init: reset subset, propagate constraints from borders
  v3i cri = sub.minCorner();
  v3i cra = sub.maxCorner();
  ForRange(k, cri[2] + 1, cra[2] - 1) {
    ForRange(j, cri[1] + 1, cra[1] - 1) {
      ForRange(i, cri[0] + 1, cra[0] - 1) {
        /*----------------------------------------------------------*/
        if (use_scene) {
          if (!S.at(i,j,k)[pal2id[ID_GROUND]] &&
              !S.at(i,j,k)[pal2id[ID_OBJECT]] &&
              !S.at(i,j,k)[pal2id[ID_CONTACT]]) {
            S.at(i, j, k).fill(true); // only original line in this section
            S.at(i, j, k).set(pal2id[ID_GROUND], false);
            S.at(i, j, k).set(pal2id[ID_OBJECT], false);
            S.at(i, j, k).set(pal2id[ID_CONTACT], false);
          }
        } else {
          S.at(i, j, k).fill(true);
        }
        /*----------------------------------------------------------*/
      }
    }
  }
  bool ok = true;
  ForRange(k, cri[2], cra[2]) {
    ForRange(i, cri[0], cra[0]) {
      ok &= propagateConstraints(i, cri[1], k, S);
      ok &= propagateConstraints(i, cra[1], k, S);
    }
    ForRange(j, cri[1], cra[1]) {
      ok &= propagateConstraints(cri[0], j, k, S);
      ok &= propagateConstraints(cra[0], j, k, S);
    }
  }
  ForRange(j, cri[1], cra[1]) {
    ForRange(i, cri[0], cra[0]) {
      ok &= propagateConstraints(i, j, cri[2], S);
      ok &= propagateConstraints(i, j, cra[2], S);
    }
  }
  return ok; // failed due to propagation
}

/* -------------------------------------------------------- */

// Counts the number of non empty labels in a sub domain (ignoring border)
int num_solids_sub(Array3D<Presence>& S, int lbl_empty, AAB<3, int> sub)
{
  int num = 0;
  v3i cri = sub.minCorner();
  v3i cra = sub.maxCorner();
  ForRange(k, cri[2] + 1, cra[2] - 1) {
    ForRange(j, cri[1] + 1, cra[1] - 1) {
      ForRange(i, cri[0] + 1, cra[0] - 1) {
        if (!S.at(i, j, k)[lbl_empty]) num++;
      }
    }
  }
  return num;
}

/* -------------------------------------------------------- */

// Main synthesis function
// Performs synthesis within the sub domain given as a box, or the full domain
// if no sub domain is specified.
// Returns true on success, false otherwise (i.e. constraints cannot be resolved).
// The domain is changed, even on failure. Caller is responsible for restoring it.
// After a success _num_solids contains the number of synthesized non empty labels.
bool synthesize(
  Array3D<Presence>& S,
  int lbl_empty, int& _num_solids,
  AAB<3, int> sub = AAB<3, int>())
{
  // buffer for choices
  int choices[1024];
  sl_assert(num_lbls < 1024);
  // box to operate upon
  AAB<3, int> box;
  if (!sub.empty()) {
    box = sub;
  } else {
    box.addPoint(v3i(0, 0, 0));
    box.addPoint(v3i(S.xsize() - 1, S.ysize() - 1, S.zsize() - 1));
  }

  // starting
  int num_choices = 0;
  _num_solids = 0;

  // randomize scanline order
  int order[] = { 0, 1, 2 };
  ForIndex(p, 9) {
    int a = rand() % 3;
    int b = rand() % 3;
    std::swap(order[a],order[b]);
  }
  v3i starts = box.minCorner();
  v3i ends   = box.maxCorner();
  int sign[] = { 1, 1, 1 };
  ForIndex(p, 3) {    // creates a random vector of 1 or -1
    sign[p] = 1 - 2 * (rand() & 1);
  }
  ForIndex(p, 3) {
    if (sign[p] < 0) {
      std::swap(starts[p], ends[p]);
    }
    ends[p] += sign[p];
  }

  // propagate until done or conflict
  v3i cur = starts;

  bool failed = false;
  while (!failed) {
    // for loop over the domain w/ random directions and starting points
    cur[order[0]] += sign[order[0]];
    if (cur[order[0]] == ends[order[0]]) {
      cur[order[0]] = starts[order[0]];
      cur[order[1]] += sign[order[1]];
      if (cur[order[1]] == ends[order[1]]) {
        cur[order[1]] = starts[order[1]];
        cur[order[2]] += sign[order[2]];
        if (cur[order[2]] == ends[order[2]]) {
          break;
        }
      }
    }

    sl_assert(cur[0] > -1 && cur[0] < (int)S.xsize());
    sl_assert(cur[1] > -1 && cur[1] < (int)S.ysize());
    sl_assert(cur[2] > -1 && cur[2] < (int)S.zsize());

    // which choices do we have here?
    num_choices = 0;
    ForIndex(l, num_lbls) {
      if (S.at(cur[0], cur[1], cur[2])[l]) {
        choices[num_choices++] = l;
      }
    }
    // failure?
    if (num_choices == 0) {
      failed = true;
    }
    // random choice
    int r = rand() % num_choices;
    int c = choices[r];
    S.at(cur[0], cur[1], cur[2]).fill(false);
    S.at(cur[0], cur[1], cur[2]).set(c,true);
    if (c != lbl_empty) {
      _num_solids ++;
    }

    // propagate this change
    bool ok = propagateConstraints(cur[0], cur[1], cur[2], S);
    if (!ok) {
      failed = true;
    }

  } // main update loop

  if (failed) {
    
    // giving up :-(
    return false;

  } else {

    // done!
    return true;
  }

}

/* -------------------------------------------------------- */

// Loads a voxel grid (.slab.vox format as exported by MagicaVoxel).
void loadFromVox(const char *fname,Array3D<uchar>& _voxels,Array<v3b>& _palette)
{
  FILE *f;
  f = fopen(fname, "rb");
  sl_assert(f != NULL);
  long sx, sy, sz;    // grid size
  fread(&sx, 4, 1, f);
  fread(&sy, 4, 1, f);
  fread(&sz, 4, 1, f);
  _voxels.allocate(sx, sy, sz);
  ForIndex(i, sx) { ForIndex(j, sy) { ForIndex(k, sz) {
        // j and k coordinates inverted to use MagicaVoxel's coord. system
        fread(&_voxels.at(i, sy - 1 - j, sz - 1 - k), sizeof(uchar), 1, f);   // read voxel label
  } } }
  _palette.allocate(256);
  fread(_palette.raw(), sizeof(v3b), 256, f);   // read palette information
  fclose(f);
}

/* -------------------------------------------------------- */

// Loads a 3D problem (.slab.vox format as exported by MagicaVoxel).
// Each voxel palette id becomes a label (renumbering is performed).
// When two voxels are neighboring in the exemplar, they are allowed 
// to appear together in the output. (What is observed is allowed,
// everything else is forbidden).
// See README.md for more details.
void load3DProblem(const char *fname)
{
  std::cout << "Constraint file information\n\n";
  // read voxels
  Array3D<uchar> grid;
  loadFromVox(fname, grid, palette);
  // build (unique) label set
  set<uchar> labels;
  ForArray3D(grid, i, j, k) {
    uchar lbl = grid.at(i, j, k);
    labels.insert(lbl);
  }

  num_lbls = (int)labels.size();    // # of different labels
  std::cout << "MVPal to id conversion: num_lbls = " << num_lbls << "\n";
  int id = 0;
  for (uchar l : labels) {    // unique id \in [0, num_lbls] for each label \in [0, ID_EMPTY]
    pal2id[l] = id;
    id2pal[id] = l;
    std::cout << "pal:\t" << (int) l + 1 << "\t<-->\tid:\t" << id << "\n"; 
    id++;
  }
  // now construct constraints
  constraints.allocate(num_lbls, num_lbls);
  constraints.fill(0);
  ForArray3D(grid, i, j, k) {   // for each voxel in the example
    int id = pal2id[grid.at(i,j,k)];
    ForIndex(n, 6) {            // for each neighbor of that voxel
      int lbl = grid.at<Wrap>(i + neighs[n][0], j + neighs[n][1], k + neighs[n][2]);
      sl_assert(pal2id.find(lbl) != pal2id.end());
      int neigh_id = pal2id[lbl];
      if (side[n]) {
        constraints.at(id, neigh_id) |= face[n];    // add constraints to pair
      } else {
        constraints.at(neigh_id, id) |= face[n];
      }
    }
  }
  // prepare table for faster constraint checks
  prepareFastConstraintChecks();

  std::cout << "\nConstraint information (MVPal)\n";
  ForIndex(l1, num_lbls) {
    std::cout << "\ncurr:\t" << (int) id2pal[l1] + 1 << "\t";
    ForIndex(n, 6) {
      if (n != 0) std::cout << "\t\t";
      std::cout << "side:\t" << dirToStr(n) << "\t" << "neig:\t";
      for(int l2 : allowed_by_side[n][l1]) {
        std::cout << (int) id2pal[l2] + 1 << ", ";
      }
      std::cout << "\n";
    }
  }

  // ready!
}

/* -------------------------------------------------------- */

// Saves a voxel file (.slab.vox format, can be imported by MagicaVoxel)
void saveAsVox(const char *fname,const Array3D<Presence>& S, Array<v3b>& _palette)
{
  FILE *f;
  f = fopen(fname, "wb");
  sl_assert(f != NULL);
  long sx = S.xsize(), sy = S.ysize(), sz = S.zsize();
  fwrite(&sx, 4, 1, f);
  fwrite(&sy, 4, 1, f);
  fwrite(&sz, 4, 1, f);
  ForIndex(i, sx) {
    ForIndex(j, sy) {
      //ForRangeReverse(k, sz-1, 0) {
      ForIndex(k, sz) {
        int id = -1;
        ForIndex(l, num_lbls) {
          // j and k coordinates inverted to use MagicaVoxel's coord. system
          if (S.at(i, sy - 1 - j, sz - 1 - k)[l]) {
            id = l;
            break;
          }
        }
        sl_assert(id > -1);
        uchar pal = id2pal[id];
        fwrite(&pal, sizeof(uchar), 1, f);
      }
    }
  }
  fwrite(_palette.raw(), sizeof(v3b), 256, f);
  fclose(f);
}

/* -------------------------------------------------------- */

// Implements model synthesis for a 3D problem
// This is using the basic building blocks above.
// The approach used here is similar to Paul Merrell's model
// synthesis: it starts empty and attempts to synthesize within
// sub-domains. This works best on difficult problems. 
// WFC is also possible by synthesizing within the entire domain.
//
// Some of the constants below (number of iterations, etc.) could
// be changed for better/faster results depending on the input problem.
// Whether everything can be determined automatically is an interesting
// (and likely difficult) question.
void solve3D()
{
  Timer tm("solve3D");

  string fullpath = string(SRC_PATH "/exemplars/") + problem + ".slab.vox";

  //// setup a 3D problem
  load3DProblem(fullpath.c_str());

  // array being synthesized
  Array3D<Presence> S;
  S.allocate(sz, sz, sz);

  /*-----------------------------------------------------------------------*/
  // Reserved labels:
  // 255 --> empty
  // 254 --> ground
  // 253 --> object
  // 252 --> contact

  if (use_scene) {
    // load the scene
    string scenepath = string(SRC_PATH "/exemplars/") + scene + ".slab.vox";
    Array3D<uchar> scenegrid;
    Array<v3b> scenepalette;

    loadFromVox(scenepath.c_str(), scenegrid, scenepalette);

    // ForArray3D(scenegrid, i, j, k) {
    //   std::cout << i << " " << j << " " << k << ";\t" << (int) scenegrid.at(i, j, k) << "\n";
    // }

    if (!init_global_scene(S, scenegrid)) {
      std::cout << "Unstable initial state...\n";
      return;
    }

    // return;

  } else {
  /*-----------------------------------------------------------------------*/
    //// init as empty 
    if (pal2id.find(ID_GROUND) != pal2id.end()) {
      // ground is being used
      init_global_empty(S, pal2id[ID_EMPTY], pal2id[ID_GROUND]);
    } else {
      // no ground: use an empty border along all faces
      init_global_empty(S, pal2id[ID_EMPTY]);
    }
  }
  //// synthesize subsets
  int num_failed    = 0;
  int num_success   = 0;
  int num_passes    = sz; // increases on larger domains.
  int num_sub_synth = 32; // will use twice that on ground level
  ForIndex(p, num_passes) {
    // ForIndex(n, p == 0 ? 2 * num_sub_synth : num_sub_synth) {
    ForIndex(n, num_sub_synth) {
      // random size
      int subsz = min(15, 8 + (rand() % 9));
      // random location
      // (forces the first pass to be on the ground, as many problems have ground constraints)
      AAB<3, int> sub;
      sub.minCorner() = v3i(
        rand() % (sz - subsz),
        rand() % (sz - subsz),
        rand() % (sz - subsz));
        //p == 0 ? 0 : rand() % (sz - subsz));
      sub.maxCorner() = sub.minCorner() + v3i(subsz, subsz, subsz);

      // sub.maxCorner() = v3i(
      //   subsz + (rand() % (sz - subsz)),
      //   subsz + (rand() % (sz - subsz)),
      //   //rand() % (sz - subsz));
      //   p == 0 ? sz - 1 : (subsz + (rand() % (sz - subsz))));
      // sub.minCorner() = sub.maxCorner() - v3i(subsz, subsz, subsz);

      // backup current
      Array3D<Presence> backup = S;
      // try reseting the subdomain (may fail)
      int num_solids_before = num_solids_sub(S, pal2id[ID_EMPTY]/*empty*/, sub);
      if (reinit_sub(S, pal2id[ID_EMPTY], sub)) {    // resets subdomain AND propagates constraints
        // try synthesizing (may fail)
        int num_solids;
        if (synthesize(S, pal2id[ID_EMPTY]/*empty*/, num_solids, sub)) {
          if (num_solids >= num_solids_before) { // only accept if less (or eq) non empty appear
            num_success++;
          } else {
            num_success++;
            // num_failed++;
            // S = backup;
          }
        } else {
          // synthesis failed: retry
          num_failed++;
          S = backup;
        }
      } else {
        // reinit failed: cannot work here 
        num_failed++;
        S = backup;
      }
    }
    // display progress
    Console::cursorGotoPreviousLineStart();
    std::cerr << sprint("attempt %3d / %3d, failures: %3d, successes: %3d\n", (p+1) * num_sub_synth, num_sub_synth*num_passes, num_failed, num_success);
  }

  // output final
  saveAsVox(SRC_PATH "/results/synthesized.slab.vox", S, palette);
}

/* -------------------------------------------------------- */

void testLoadSave() {
  string scenepath = string(SRC_PATH "/exemplars/") + scene + ".slab.vox";
  Array3D<uchar> scenegrid;
  Array<v3b> scenepalette;

  loadFromVox(scenepath.c_str(), scenegrid, scenepalette);

  // ForArray3D(scenegrid, i, j, k) {
  //   if ((int) scenegrid.at(i, j, k) != ID_EMPTY)
  //     std::cout << "(" << i << ", " << j << ", " << k << "):\t" << (int) scenegrid.at(i, j, k) << "\n";
  // }

  set<uchar> labels;
  ForArray3D(scenegrid, i, j, k) {
    uchar lbl = scenegrid.at(i, j, k);
    labels.insert(lbl);
  }
  num_lbls = (int)labels.size();    // # of different labels
  int id = 0;
  for (uchar l : labels) {    // unique id \in [0, num_lbls] for each label \in [0, ID_EMPTY]
    std::cout << (int) l << std::endl;
    pal2id[l] = id;
    id2pal[id] = l;
    id++;
  }

  Array3D<Presence> S;
  S.allocate(sz, sz, sz);
  init_only_scene(S, scenegrid);
  
  saveAsVox(SRC_PATH "/results/init.slab.vox", S, scenepalette);
}

/* -------------------------------------------------------- */

// This is where it all begins.
int main(int argc, char **argv) 
{
  try {

    // random seed
    srand((unsigned int)time(NULL));
    
    // let's synthesize!
    std::cerr << Console::white << "Synthesizing a voxel model!" << Console::gray << std::endl << std::endl;
    solve3D();
    //testLoadSave();

  } catch (Fatal& e) {
    std::cerr << Console::red << e.message() << Console::gray << std::endl;
    return (-1);
  }

  return (0);
}

/* -------------------------------------------------------- */
