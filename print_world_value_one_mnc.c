#include  <stdio.h>
#include  <volume_io/internal_volume_io.h>
#include  <bicpl.h>

#define EVAL_NEAREST -1
#define EVAL_LINEAR 0
#define EVAL_CUBIC 2

int interpolant = EVAL_LINEAR;

int  main(
    int   argc,
    char  *argv[] )
{
    STRING     input_filename, coordlist_filename;
    FILE*      coordfile;
    //Real       x, y, z, value;
    double     *x, *y, *z;
    double     value;
    double     curx, cury, curz;
    int        voxx, voxy, voxz, sizes[MAX_DIMENSIONS];
    Real       voxel[MAX_DIMENSIONS];
    int i, keep_looping, n_coords;
    Volume     volume;

    initialize_argument_processing( argc, argv );

    if( !get_string_argument( "", &input_filename ) )
    {
        return( 1 );
    }

    if( !get_string_argument( "", &coordlist_filename ) ) {
        return( 1 );
    }

    /* first pass: count the number of coordinates in the list */
    coordfile = fopen(coordlist_filename, "r+t");
    n_coords = 0;
    keep_looping = 1;
    while(keep_looping) {
        if(fscanf(coordfile, "%lf%lf%lf", &curx, &cury, &curz) != 3) {
            keep_looping = 0;
        } else {
            n_coords++;
        }
    }
    fclose(coordfile);

    x = (double*)malloc( n_coords * sizeof( double ) );
    y = (double*)malloc( n_coords * sizeof( double ) );
    z = (double*)malloc( n_coords * sizeof( double ) );
    if( !x || !y || !z ) {
        printf( "Failed to allocate memory for %d coordinates.\n", 
            n_coords );
        return(1);
    }

    /* read coordlist into x/y/z arrays */
    coordfile = fopen(coordlist_filename, "r+t");
    i = 0;
    keep_looping = 1;
    while(keep_looping && i < n_coords) {
        curx = 0;
        cury = 0;
        curz = 0;
        if(fscanf(coordfile, "%lf%lf%lf", &curx, &cury, &curz) != 3) {
            keep_looping = 0;
        } else {
            x[i] = curx;
            y[i] = cury;
            z[i] = curz;
            ++i;
        }
    }
        fclose(coordfile);


 
    if( input_volume( input_filename, 3, XYZ_dimension_names,
            NC_UNSPECIFIED, FALSE, 0.0, 0.0,
            TRUE, &volume, (minc_input_options *) NULL ) != OK )
        return( 1 );

    for(i = 0; i < n_coords ; ++i) {
        get_volume_sizes( volume, sizes );
        //convert_world_to_voxel(volume, x[i], y[i], z[i], voxel);

        voxx = FLOOR( voxel[0] );
        voxy = FLOOR( voxel[1] );
        voxz = FLOOR( voxel[2] );

        if( voxx >= 0 && voxx < sizes[0] &&
            voxy >= 0 && voxy < sizes[1] &&
            voxz >= 0 && voxz < sizes[2] )
        {

            evaluate_volume_in_world( volume, x[i], y[i], z[i], interpolant, FALSE, 0.0, &value, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL );
            // value = (double) get_volume_real_value( volume, voxx, voxy, voxz, 0, 0);
            printf( "%lf\t", value );
        } else {
            //printf("Voxel %d %d %d is outside %s\n", voxx, voxy, voxz, cur_minc);
            printf("NaN\t");
        }
                        
    }
    delete_volume(volume);

    free( x );
    free( y );
    free( z );

    return( 0 );
}
