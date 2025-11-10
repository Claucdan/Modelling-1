#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
using namespace Eigen;

const double HBAR2_OVER_2M = 0.0380998;

double a = 2.0;
int N = 800;
int num_states = 8;
string potential_input_file = "potential_input.dat";

enum class PotentialType { RECT_WELL, PARABOLIC, TRIANGULAR, FROM_FILE };

struct Params {
  double depth = 5.0;
  double width = 1.0;
  double center = 1.0;
  double k = 5.0;
} P;

double U_rect(double x) {
  double left = P.center - P.width / 2.0;
  double right = P.center + P.width / 2.0;
  if (x >= left && x <= right)
    return -P.depth;
  return 0.0;
}

double U_parabolic(double x) { return P.k * pow(x - P.center, 2.0) - P.depth; }

double U_triangular(double x) {
  double left = P.center - P.width / 2.0;
  double right = P.center + P.width / 2.0;
  if (x < left || x > right)
    return 0.0;
  double mid = P.center;
  if (x <= mid)
    return -P.depth * (x - left) / (mid - left);
  return -P.depth * (right - x) / (right - mid);
}

bool read_potential_from_file(const string &filename, vector<double> &xs,
                              vector<double> &Us) {
  ifstream fin(filename);
  if (!fin)
    return false;
  string line;
  xs.clear();
  Us.clear();
  while (getline(fin, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    double x, U;
    stringstream ss(line);
    if (ss >> x >> U) {
      xs.push_back(x);
      Us.push_back(U);
    }
  }
  return !xs.empty();
}

double interp_linear(const vector<double> &xs, const vector<double> &Us,
                     double x) {
  if (xs.empty())
    return 0.0;
  if (x <= xs.front())
    return Us.front();
  if (x >= xs.back())
    return Us.back();
  auto it = upper_bound(xs.begin(), xs.end(), x);
  size_t i = it - xs.begin() - 1;
  double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
  return Us[i] + t * (Us[i + 1] - Us[i]);
}

int main() {
  cout << fixed << setprecision(3);
  cout << "=== Моделирование уравнения Шрёдингера 1D ===\n";
  cout << "Введите ширину области a (нм): ";
  cin >> a;
  cout << "Введите число точек сетки N: ";
  cin >> N;
  cout << "Введите, сколько уровней вывести: ";
  cin >> num_states;

  cout << "\nВыберите тип потенциала:\n";
  cout << " 1 — Прямоугольная яма\n";
  cout << " 2 — Параболическая яма\n";
  cout << " 3 — Треугольная яма\n";
  cout << " 4 — Таблично из файла potential_input.dat\n";
  cout << "Ваш выбор: ";
  int choice;
  cin >> choice;

  PotentialType type;
  switch (choice) {
  case 1:
    type = PotentialType::RECT_WELL;
    break;
  case 2:
    type = PotentialType::PARABOLIC;
    break;
  case 3:
    type = PotentialType::TRIANGULAR;
    break;
  case 4:
    type = PotentialType::FROM_FILE;
    break;
  default:
    type = PotentialType::RECT_WELL;
  }

  if (type != PotentialType::FROM_FILE) {
    cout << "Глубина ямы (эВ): ";
    cin >> P.depth;
    cout << "Ширина ямы (нм): ";
    cin >> P.width;
    cout << "Центр ямы (нм): ";
    cin >> P.center;
    if (type == PotentialType::PARABOLIC) {
      cout << "Коэффициент параболы k (эВ/нм²): ";
      cin >> P.k;
    }
  }

  double dx = a / (N + 1.0);
  vector<double> xs_file, Us_file;
  bool have_file = false;

  if (type == PotentialType::FROM_FILE) {
    have_file =
        read_potential_from_file(potential_input_file, xs_file, Us_file);
    if (!have_file) {
      cerr << "Ошибка: файл " << potential_input_file
           << " не найден или пуст.\n";
      return 1;
    }
  }

  VectorXd Uvec(N);
  for (int i = 0; i < N; ++i) {
    double x = (i + 1) * dx;
    switch (type) {
    case PotentialType::RECT_WELL:
      Uvec(i) = U_rect(x);
      break;
    case PotentialType::PARABOLIC:
      Uvec(i) = U_parabolic(x);
      break;
    case PotentialType::TRIANGULAR:
      Uvec(i) = U_triangular(x);
      break;
    case PotentialType::FROM_FILE:
      Uvec(i) = interp_linear(xs_file, Us_file, x);
      break;
    }
  }

  ofstream fU("potential.dat");
  fU << "# x (нм)\tU(x) (эВ)\n";
  for (int i = 0; i < N; ++i)
    fU << (i + 1) * dx << "\t" << Uvec(i) << "\n";
  fU.close();

  double coef = HBAR2_OVER_2M / (dx * dx);
  MatrixXd H = MatrixXd::Zero(N, N);
  for (int i = 0; i < N; ++i) {
    H(i, i) = 2.0 * coef + Uvec(i);
    if (i > 0)
      H(i, i - 1) = -coef;
    if (i < N - 1)
      H(i, i + 1) = -coef;
  }

  cout << "\nДиагонализация матрицы " << N << "x" << N << " ...\n";
  SelfAdjointEigenSolver<MatrixXd> es(H);
  if (es.info() != Success) {
    cerr << "Ошибка диагонализации!\n";
    return 1;
  }

  VectorXd E = es.eigenvalues();
  MatrixXd psi = es.eigenvectors();
  int nsave = min(num_states, N);

  for (int n = 0; n < nsave; ++n) {
    double norm = sqrt(psi.col(n).squaredNorm() * dx);
    psi.col(n) /= norm;
  }

  double max_abs = psi.array().abs().maxCoeff();
  double typical_dE = (nsave > 1) ? (E(nsave - 1) - E(0)) / (nsave - 1) : 1.0;
  double amp_scale = 0.3 * typical_dE / max_abs;

  ofstream fw("wavefunctions.dat");
  fw << "# x (нм)\tU(x)\t";
  for (int n = 0; n < nsave; ++n)
    fw << "E" << n + 1 << "\tpsi" << n + 1 << "+E\t";
  fw << "\n";

  for (int i = 0; i < N; ++i) {
    double x = (i + 1) * dx;
    fw << x << "\t" << Uvec(i);
    for (int n = 0; n < nsave; ++n)
      fw << "\t" << E(n) << "\t" << (E(n) + amp_scale * psi(i, n));
    fw << "\n";
  }
  fw.close();

  ofstream gp("plot.gp");
  gp << fixed << setprecision(3);
  gp << "set terminal qt size 1200,700 font 'Arial,12'\n";
  gp << "set title sprintf('Собственные функции и уровни в ящике [0, %.2f] "
        "нм', "
     << a << ")\n";
  gp << "set xlabel 'x, нм'\n";
  gp << "set ylabel 'Энергия, эВ'\n";
  gp << "set grid\n";
  gp << "set key top right box spacing 1.1 font ',10'\n";
  gp << "set style line 1 lc rgb '#000000' lw 2.5\n"; // U(x)

  vector<string> colors = {"#0000FF", "#FF8C00", "#008000", "#FF0000",
                           "#800080", "#FF69B4", "#A0522D", "#808080"};

  gp << "plot \\\n";
  gp << " 'wavefunctions.dat' u 1:2 w l ls 1 title 'Потенциал U(x)'";

  for (int n = 0; n < nsave; ++n) {
    int col_E = 3 + 2 * n;
    int col_psi = 4 + 2 * n;
    string color = colors[n % colors.size()];

    stringstream title;
    title << "E_" << (n + 1) << " = " << fixed << setprecision(3) << E(n)
          << " эВ";

    gp << ", \\\n";
    gp << " '' u 1:" << col_psi << " w l lc rgb '" << color << "' lw 2 title '"
       << title.str() << "'";
    gp << ", \\\n";
    gp << " '' u 1:" << col_E << " w l dt 2 lc rgb '" << color
       << "' lw 1.2 notitle";
  }

  gp << "\n";
  gp << "pause -1 'Нажмите Enter для завершения'\n";
  gp.close();
  return 0;
}
