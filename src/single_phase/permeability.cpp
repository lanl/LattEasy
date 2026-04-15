/* Modified from file by Wim Degruyter */

#include "palabos3D.h"
#include "palabos3D.hh"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace plb;
using namespace std;

using namespace plb;

typedef double T;
#define DESCRIPTOR descriptors::D3Q19Descriptor

bool isToolOnPath(const std::string& toolName)
{
  const char* pathEnv = std::getenv("PATH");
  if (!pathEnv) {
    return false;
  }

#ifdef _WIN32
  const char pathSeparator = ';';
  const char dirSeparator = '\\';
  std::vector<std::string> suffixes = {"", ".exe", ".bat", ".cmd"};
#else
  const char pathSeparator = ':';
  const char dirSeparator = '/';
  std::vector<std::string> suffixes = {""};
#endif

  std::stringstream pathStream(pathEnv);
  std::string directory;
  while (std::getline(pathStream, directory, pathSeparator)) {
    if (directory.empty()) {
      continue;
    }
    for (std::vector<std::string>::const_iterator it = suffixes.begin(); it != suffixes.end(); ++it) {
      std::ifstream candidate((directory + dirSeparator + toolName + *it).c_str());
      if (candidate.good()) {
        return true;
      }
    }
  }
  return false;
}


// This function object returns a zero velocity, and a pressure which decreases
//   linearly in x-direction. It is used to initialize the particle populations.
class PressureGradient {
public:
  PressureGradient(T deltaP_, plint nx_) : deltaP(deltaP_), nx(nx_)
  { }
  void operator() (plint iX, plint iY, plint iZ, T& density, Array<T,3>& velocity) const
  {
    velocity.resetToZero();
    density = (T)1 - deltaP*DESCRIPTOR<T>::invCs2 / (T)(nx-1) * (T)iX;
  }
private:
  T deltaP;
  plint nx;
};

// This function grabs the appropiate geometry for the single-phase simulation
void readGeometry(
  const std::string& inputFolder,
  MultiScalarField3D<int>& geometry,
  bool vtk_out,
  const std::string& geometryName)
{
  const plint nx = geometry.getNx();
  const plint ny = geometry.getNy();
  const plint nz = geometry.getNz();
  const std::string geometryPath = inputFolder + geometryName + ".dat";

  Box3D sliceBox(0,0, 0,ny-1, 0,nz-1);

  pcout << "\nRunning absolute permeability " << std::endl;
  pcout << "The geometry name is  " << geometryPath << std::endl;

  std::unique_ptr<plb::MultiScalarField3D<int> > slice = generateMultiScalarField<int>(geometry, sliceBox);
  plb_ifstream geometryFile(geometryPath.c_str());

  for (plint iX=0; iX<nx-1; ++iX) {
    if (!geometryFile.is_open()) {
      pcout << "Error: could not open the geometry file " << geometryPath << std::endl;
      exit(EXIT_FAILURE);
    }

    geometryFile >> *slice;
    copy(*slice, slice->getBoundingBox(), geometry, Box3D(iX,iX, 0,ny-1, 0,nz-1));
  }

  if  (vtk_out == true) {
    VtkImageOutput3D<T> vtkOut(createFileName("PorousMedium", 1, 6), 1.0);
    vtkOut.writeData<float>(*copyConvert<int,T>(geometry, geometry.getBoundingBox()), "tag", 1.0);
  }


 // code to create .st file. Uncomment if needed
  //{
    //std::auto_ptr<MultiScalarField3D<T> > floatTags = copyConvert<int,T>(geometry, geometry.getBoundingBox());
    //std::vector<T> isoLevels;
    //isoLevels.push_back(0.5);
    //typedef TriangleSet<T>::Triangle Triangle;
    //std::vector<Triangle> triangles;
    //Box3D domain = floatTags->getBoundingBox().enlarge(-1);
    //domain.x0++;
    //domain.x1--;
    //isoSurfaceMarchingCube(triangles, *floatTags, isoLevels, domain);
    //TriangleSet<T> set(triangles);
    //std::string outDir = fNameOut + "/";
    //set.writeBinarySTL(outDir + "porousMedium.stl");
  //}
}

void porousMediaSetup(MultiBlockLattice3D<T,DESCRIPTOR>& lattice,
  OnLatticeBoundaryCondition3D<T,DESCRIPTOR>* boundaryCondition,
  MultiScalarField3D<int>& geometry, T deltaP)
  {
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();

    pcout << "Definition of inlet/outlet." << std::endl;

    Box3D inlet (0,0, 1,ny-2, 1,nz-2);
    boundaryCondition->addPressureBoundary0N(inlet, lattice);
    setBoundaryDensity(lattice, inlet, (T) 1.);

    Box3D outlet(nx-1,nx-1, 1,ny-2, 1,nz-2);
    boundaryCondition->addPressureBoundary0P(outlet, lattice);
    setBoundaryDensity(lattice, outlet, (T) 1. - deltaP*DESCRIPTOR<T>::invCs2);

    // Where "geometry" evaluates to 1, use bounce-back.
    defineDynamics(lattice, geometry, new BounceBack<T,DESCRIPTOR>(), 1);
    // Where "geometry" evaluates to 2, use no-dynamics (which does nothing).
    defineDynamics(lattice, geometry, new NoDynamics<T,DESCRIPTOR>(), 2);

    //   pcout << "Initialization of rho and u." << std::endl;
    initializeAtEquilibrium( lattice, lattice.getBoundingBox(),
                                PressureGradient(deltaP, nx) );

    lattice.initialize();
    delete boundaryCondition;
  }

  void writeGifs(MultiBlockLattice3D<T,DESCRIPTOR>& lattice, plint iter, plint run)
  {
    const plint nx = lattice.getNx();
    const plint ny = lattice.getNy();
    const plint nz = lattice.getNz();

    const plint imSize = 600;
    ImageWriter<T> imageWriter("leeloo");

    // Write velocity-norm at x=1.
    imageWriter.writeScaledGif(createFileName("ux_inlet", run, 6),
    *computeVelocityNorm(lattice, Box3D(2,2, 0,ny-1, 0,nz-1)),
    imSize, imSize );

    // Write velocity-norm at x=nx/2.
    imageWriter.writeScaledGif(createFileName("ux_half", run, 6),
    *computeVelocityNorm(lattice, Box3D(nx/2,nx/2, 0,ny-1, 0,nz-1)),
    imSize, imSize );
  }

  void writeVTK(MultiBlockLattice3D<T,DESCRIPTOR>& lattice, plint iter, plint run)
  {
    VtkImageOutput3D<T> vtkOut(createFileName("vtk_vel", run, 6), 1.);
    vtkOut.writeData<float>(*computeVelocityNorm(lattice), "velocityNorm", 1.);
    vtkOut.writeData<3,float>(*computeVelocity(lattice), "velocity", 1.);
  }

	  void computePermeability(
	    MultiBlockLattice3D<T,DESCRIPTOR>& lattice, T nu, T deltaP, Box3D domain,
	    T& perm, T& meanU, bool printPerm = true)
	  {

    // Compute only the x-direction of the velocity (direction of the flow).
    plint xComponent = 0;
    plint nx = lattice.getNx();
    plint ny = lattice.getNy();
    plint nz = lattice.getNz();
    Box3D domain1(0, nx-1, 0, ny-1, 0, nz-1);

	    meanU = computeAverage(*computeVelocityComponent(lattice, domain1, xComponent));

		    pcout << "Average velocity [l.u.]     = " << meanU                    << std::endl;
		    //pcout << "Lattice viscosity nu = " << nu                       << std::endl;
		    //pcout << "Grad P               = " << deltaP/(T)(nx-1)         << std::endl;
		    perm = nu*meanU / (deltaP/(T)(nx-1));
		    if (printPerm) {
		      pcout << "Permeability [l.u.^2] = " << perm << std::endl;
		    }
		    //  return meanU;
	  }

  int main(int argc, char **argv)
  {
    plbInit(&argc, &argv);


    std::string fNameIn ;
    std::string fNameOut;

	    plint nx;
	    plint ny;
	    plint nz;
	    T deltaP ;
	    plint requestedRuns;
		  bool nx_p, ny_p, nz_p;
		  bool vtk_out;
	    std::string GeometryName ;
	    plint maxT;
		  T conv;

    std::string xmlFname;
    try {
      global::argv(1).read(xmlFname);
    }
    catch (PlbIOException& exception) {
      pcout << "Wrong parameters; the syntax is: "
      << (std::string) global::argv(0) << " input-file.xml" << std::endl;
      return -1;
    }

    // 2. Read input parameters from the XML file.
    pcout << "Reading inputs from xml file \n";
    try {
      XMLreader document(xmlFname);
      document["geometry"]["file_geom"].read(GeometryName);

      document["geometry"]["size"]["x"].read(nx);
      document["geometry"]["size"]["y"].read(ny);
      document["geometry"]["size"]["z"].read(nz);
	    document["geometry"]["per"]["x"].read(nx_p);
      document["geometry"]["per"]["y"].read(ny_p);
      document["geometry"]["per"]["z"].read(nz_p);


	      document["folder"]["out_f"].read(fNameOut);
	      document["folder"]["in_f"].read(fNameIn);

	      document["simulations"]["press"].read(deltaP);
	      document["simulations"]["num"].read(requestedRuns);
	      document["simulations"]["iter"].read(maxT);
		    document["simulations"]["conv"].read(conv);
		    document["simulations"]["vtk_out"].read(vtk_out);

    }
    catch (PlbIOException& exception) {
      pcout << exception.what() << std::endl;
      pcout << exception.what() << std::endl;
      return -1;
    }


	    std::string inputF= fNameIn;
	    global::directories().setOutputDir(fNameOut+"/");
	    global::directories().setInputDir(inputF+"/");

	    if (requestedRuns != 1) {
	      pcout << "This single-phase solver supports exactly one simulation per input file. "
	            << "Set <num> to 1." << std::endl;
	      return -1;
	    }

	    const plint run = 1;
	    const T omega = 1.0;
	    const T nu    = ((T)1/omega- (T)0.5)/DESCRIPTOR<T>::invCs2;
	    T permeability = (T)0;
	    T meanVelocity = (T)0;
    pcout << "Total simulations: 1" << std::endl;
    pcout << "The convergence threshold is: " << conv << " %" << std::endl;


	    const bool gifOutputEnabled = isToolOnPath("convert");
	    if (!gifOutputEnabled) {
	      pcout << "Skipping GIF previews because ImageMagick `convert` was not found." << std::endl;
	    }

      MultiBlockLattice3D<T,DESCRIPTOR> lattice( nx,ny,nz,
                                        new BGKdynamics<T,DESCRIPTOR>(omega) );
      // Switch off periodicity.
      //lattice.periodicity().toggleAll(false);

      lattice.periodicity().toggle(0, nx_p);
      lattice.periodicity().toggle(1, ny_p);
      lattice.periodicity().toggle(2, nz_p);

      MultiScalarField3D<int> geometry(nx,ny,nz);
      readGeometry(fNameIn, geometry, vtk_out, GeometryName);

      porousMediaSetup(lattice, createLocalBoundaryCondition3D<T,DESCRIPTOR>(),
                       geometry, deltaP);


      pcout << "Simulation begins" << std::endl;
      plint iT=0;
      T new_avg_f, old_avg_f, relE_f1;
      lattice.toggleInternalStatistics(false);

      for (;iT<maxT; ++iT) {


        if (iT % 250 == 0 && iT > 0) {

          lattice.toggleInternalStatistics(true);
          pcout << "Iteration " << iT   << std::endl;
          pcout << "-----------------"  << std::endl;
          lattice.collideAndStream();
          new_avg_f = getStoredAverageEnergy(lattice);
          lattice.toggleInternalStatistics(false);
          relE_f1 = std::fabs(old_avg_f-new_avg_f)*100/old_avg_f;
          pcout << "Relative difference of Energy: " << setprecision(3)
          << relE_f1 <<" %"<<std::endl;
          pcout << "The preliminary permeability is: " <<std::endl;
	      computePermeability(lattice, nu, deltaP, lattice.getBoundingBox(), permeability, meanVelocity);
          pcout << "**********************************************" <<std::endl;
          if ( relE_f1<conv ){
            break;
          }
          old_avg_f = new_avg_f; // store new properties
        }
      }

      pcout << "End of simulation at iteration " << iT << std::endl;

      //   pcout << "Permeability:" << std::endl;
	      computePermeability(lattice, nu, deltaP, lattice.getBoundingBox(), permeability, meanVelocity, false);

		    if (gifOutputEnabled) {
		      writeGifs(lattice,iT,run);
		    }
	    std::string outDir = fNameOut + "/";
      std::string vel_name = outDir + GeometryName + "_vel.dat";
      plb_ofstream ofile3( vel_name.c_str() );
      ofile3 << setprecision(10) <<*computeVelocity(lattice) << endl;

      std::string rho_name = outDir + GeometryName + "_rho.dat";
      plb_ofstream ofile4( rho_name.c_str() );
      ofile4 << setprecision(10) <<*computeDensity(lattice) << endl;

      pcout << "Absolute Permeability [l.u.^2] = " << permeability << std::endl;

	 if  (vtk_out == true) {
      pcout << "Writing VTK file ..." << std::endl;
      writeVTK(lattice, iT, run);
	 }

	    pcout << "Printing outputs" << std::endl;
	    std::string output = outDir + "relPerm&vels.txt";
	    plb_ofstream ofile(output.c_str());
	    ofile << "Outputs" << "\n\n";
	    ofile << "Absolute Permeability [l.u.^2]   = " << permeability << std::endl;
	    ofile << "Mean Velocity [l.u.]   = " << meanVelocity << std::endl;
	  }
