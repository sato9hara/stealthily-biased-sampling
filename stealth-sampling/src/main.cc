//
// Labels must be nonnegative integers 
//


#include <iostream>
#include <fstream>
#include <lemon/smart_graph.h>
#include <lemon/network_simplex.h>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;
using namespace lemon;

// User must define feature and distance function
using MyFeature = vector<pair<int, double>>; // = sparse vector
struct MyDatum {
  int label;
  MyFeature feature;
};
double distance(MyFeature x, MyFeature y) {
  auto square = [&](double t) { return t * t; };
  double score = 0;
  for (int i = 0, j = 0; i != x.size() || j != y.size(); ) {
    if (i == x.size()) {
      score += square(y[j].second);
      ++j;
    } else if (j == y.size()) {
      score += square(x[i].second);
      ++i;
    } else if (x[i].first < y[j].first) {
      score += square(x[i].second);
      ++i;
    } else if (x[i].first > y[j].first) {
      score += square(y[j].second);
      ++j;
    } else {
      score += square(x[i].second - y[j].second);
      ++i;
      ++j;
    } 
  }
  return score;
}
vector<MyDatum> readFile(string filename) {
  vector<MyDatum> data;
  ifstream ifs(filename);
  
  for (string line; getline(ifs, line); ) {
    if (line[0] == '#') continue;
    MyDatum datum;
    for (int i = 0; i < line.size(); ++i) 
      if (line[i] == ':') line[i] = ' ';
    stringstream ss(line);
    ss >> datum.label;
    int index;
    double value;
    while (ss >> index >> value) {
      datum.feature.push_back(make_pair(index, value));
    }
    sort(datum.feature.begin(), datum.feature.end());
    data.push_back(datum);
  }
  return data;
}
vector<int> readK(string filename) {
  vector<int> K;
  ifstream ifs(filename);
  int skip = 0;

  for (string line; getline(ifs, line); ) {
    if (line[0] == '#') {
      skip++;
      if (skip == 4) {
        stringstream ss(line.substr(2, line.size()-2));
        int k;
        while (ss >> k) {
          K.push_back(k);
        }
      }
    }
  }
  return K;
}

// NN Descent
template <class D>
struct NNDescent {
  size_t n, k;
  typedef pair<size_t, double> node;
  vector<vector<node>> adj;
  D d;
  NNDescent(size_t n, size_t k, D d) : n(n), k(k), adj(n), d(d) {
    for (size_t i = 0; i < n; ++i) {
      while (adj[i].size() < k) {
        size_t j = rand() % n;
        if (i != j) adj[i].push_back({j, d(i,j)});
      }
    }
    for (int iter = 0; iter < 20 && update(); ++iter);
  }
  bool update() {
    vector<unordered_map<size_t, double>> nbh(n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t a = 0; a < adj[i].size(); ++a) {
        size_t j = adj[i][a].first;
        if (!nbh[i].count(j)) 
          nbh[i][j] = nbh[j][i] = adj[i][a].second;
        for (size_t b = 0; b < a; ++b) {
          size_t k = adj[i][b].first;
          if (!nbh[j].count(k)) 
            nbh[j][k] = nbh[k][j] = d(j, k);
        }
      }
    }
    bool cont = false;
    for (size_t i = 0; i < n; ++i) {
      vector<node> knbh(nbh[i].begin(), nbh[i].end());
      partial_sort(knbh.begin(), knbh.begin()+k, knbh.end(),
          [](const node &a, const node &b) { return a.second < b.second; });
      for (size_t j = 0; j < k; ++j) {
        if (adj[i][j].second != knbh[j].second) cont = true;
        adj[i][j] = knbh[j];
      }
    }
    return cont;
  }
  vector<int> nbh(int i) {
    vector<int> res;
    for (auto z: adj[i]) {
      res.push_back(z.first);
    }
    return res;
  }
};
template <class D>
NNDescent<D> getNNDescent(int n, int k, D d) { return NNDescent<D>(n, k, d); }


// sizes[label] = number of points having the label
template <class Datum>
vector<double> biasedSampling(vector<Datum> data, vector<int> sizes) {
  int K = 0, N = data.size();
  for (int i = 0; i < sizes.size(); ++i)
    K += sizes[i];
  int demand = N * K;

  SmartDigraph g;
  SmartDigraph::ArcMap<int> capacity(g);
  SmartDigraph::ArcMap<double> cost(g);

  // supersource, superterminal
  SmartDigraph::Node s = g.addNode();
  SmartDigraph::Node t = g.addNode();

  // intermediate vertices to control class biases
  vector<SmartDigraph::Node> intermediate;
  for (int i = 0; i < sizes.size(); ++i) {
    SmartDigraph::Node u = g.addNode();
    intermediate.push_back(u);
    SmartDigraph::Arc a = g.addArc(s, u);
    capacity[a] = N * sizes[i];
    cost[a] = 0;
  }

  // left vertices
  vector<SmartDigraph::Node> left;
  vector<SmartDigraph::Arc> incomming;
  for (int i = 0; i < data.size(); ++i) {
    left.push_back(g.addNode());
    SmartDigraph::Node u = intermediate[data[i].label];
    SmartDigraph::Arc a = g.addArc(u, left[i]);
    capacity[a] = N; 
    cost[a] = 0;
    incomming.push_back(a);
  }
  // right vertices
  vector<SmartDigraph::Node> right;
  for (int i = 0; i < data.size(); ++i) {
    right.push_back(g.addNode());
    SmartDigraph::Arc a = g.addArc(right[i], t);
    capacity[a] = K; 
    cost[a] = 0;
  }

  // k neighborhood graph
  /*
  auto d = [&](int i, int j) {
    return distance(data[i].feature, data[j].feature);
  };
  */
  //auto nng = getNNDescent(N, 300, d);

  for (int i = 0; i < left.size(); ++i) {
    /*
    for (int j: nng.nbh(i)) {
      SmartDigraph::Arc a = g.addArc(left[i], right[j]);
      capacity[a] = N * K; 
      cost[a] = distance(data[i].feature, data[j].feature);
    }
    */
    for (int j = 0; j < right.size(); ++j) {
      SmartDigraph::Arc a = g.addArc(left[i], right[j]);
      capacity[a] = N * K; 
      cost[a] = distance(data[i].feature, data[j].feature);
    }
  }

  NetworkSimplex<SmartDigraph, int, double> ns(g);
  ns.upperMap(capacity);
  ns.costMap(cost);
  ns.stSupply(s, t, N*K);

  bool res = ns.run();
  if (!res) cerr << "infeasible" << endl;
  SmartDigraph::ArcMap<int> flow(g);
  ns.flowMap(flow);

  vector<double> weight(N);
  for (int i = 0; i < N; ++i) {
    weight[i] = flow[incomming[i]];
    cout << weight[i] << endl;
  }
  //cerr << ns.totalCost() << endl;
  return weight;
}


int main(int argc, char *argv[]) {
  vector<MyDatum> data = readFile(argv[1]);
  //vector<int> sizes = {30, 10, 10, 30};
  vector<int> sizes = readK(argv[1]);
  vector<double> weight = biasedSampling(data, sizes);
}
