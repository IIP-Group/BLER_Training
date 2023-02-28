import tensorflow as tf
from sionna.channel import set_3gpp_scenario_parameters
from sionna.channel.utils import generate_uts_topology
from sionna import PI

# This file modifies the Sionna function `gen_single_sector_topology` such that the base station antenna faces the
# sector (cell) center. We modified bs_yaw = PI/3, which will be part of Sionna release v0.13.

def gen_single_sector_topology( batch_size,
                                num_ut,
                                scenario,
                                min_bs_ut_dist=None,
                                isd=None,
                                bs_height=None,
                                min_ut_height=None,
                                max_ut_height=None,
                                indoor_probability = None,
                                min_ut_velocity=None,
                                max_ut_velocity=None,
                                dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin and ``num_ut`` UTs randomly and uniformly dropped in a cell sector.

    The following picture shows the sector from which UTs are sampled.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    UTs orientations are randomly and uniformly set, whereas the BS orientation
    is set such that the it is oriented towards the center of the sector.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.channel.tr38901.UMi`, :class:`~sionna.channel.tr38901.UMa`,
    and :class:`~sionna.channel.tr38901.RMa`.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology(batch_size = 100,
    ...                                       num_ut = 4,
    ...                                       scenario = 'umi')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology.png

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    scenario : str
        System leven model scenario. Must be one of "rma", "umi", or "uma".

    min_bs_ut_dist : None or tf.float
        Minimum BS-UT distance [m]

    isd : None or tf.float
        Inter-site distance [m]

    bs_height : None or tf.float
        BS elevation [m]

    min_ut_height : None or tf.float
        Minimum UT elevation [m]

    max_ut_height : None or tf.float
        Maximum UT elevation [m]

    indoor_probability : None or tf.float
        Probability of a UT to be indoor

    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or tf.float
        Maximim UT velocity [m/s]

    dtype : tf.DType
        Datatype to use for internal processing and output.
        If a complex datatype is provided, the corresponding precision of
        real components is used.
        Defaults to `tf.complex64` (`tf.float32`).

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], tf.float
        UTs locations

    bs_loc : [batch_size, 1, 3], tf.float
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], tf.float
        UTs orientations [radian]

    bs_orientations : [batch_size, 1, 3], tf.float
        BS orientations [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], tf.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], tf.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """

    params = set_3gpp_scenario_parameters(  scenario,
                                            min_bs_ut_dist,
                                            isd,
                                            bs_height,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            dtype)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity = params

    real_dtype = dtype.real_dtype

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack([ tf.zeros([batch_size, 1], real_dtype),
                        tf.zeros([batch_size, 1], real_dtype),
                        tf.fill( [batch_size, 1], bs_height)], axis=-1)

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist + 0.5*isd)*0.5
    bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    bs_yaw = tf.constant(PI/3.0, real_dtype)       # now points to cell center (120 degrees sectors)
    bs_orientation = tf.stack([ tf.fill([batch_size, 1], bs_yaw),
                                tf.fill([batch_size, 1], bs_downtilt),
                                tf.zeros([batch_size, 1], real_dtype)], axis=-1)

    # Generating the UTs
    ut_topology = generate_uts_topology(    batch_size,
                                            num_ut,
                                            'sector',
                                            tf.zeros([2], real_dtype),
                                            min_bs_ut_dist,
                                            isd,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            dtype)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities,\
            in_state
