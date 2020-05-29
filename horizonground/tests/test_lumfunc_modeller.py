import pytest
from astropy.cosmology import Planck15

from horizonground.lumfunc_modeller import (
    LumFuncModeller,
    alpha_emitter_schechter_lumfunc,
    quasar_hybrid_lumfunc,
    quasar_hybrid_model_constraint,
    quasar_PLE_lumfunc,
    quasar_PLE_model_constraint,
)


@pytest.mark.parametrize(
    "m,z,lg_phi",
    [
        (-27., 1., -7.99),
        (-26., 2., -5.92),
    ]
)
def test_quasar_PLE_lumfunc(quasar_PLE_model_params, m, z, lg_phi):

    assert quasar_PLE_lumfunc(m, z, **quasar_PLE_model_params) \
        == pytest.approx(lg_phi, abs=0.01)


@pytest.mark.parametrize(
    "m,z,lg_phi",
    [
        (-27., 1., -7.88),
        (-26., 2., -5.92),
        (-25., 3.25, -6.22),
    ]
)
def test_quasar_hybrid_lumfunc(quasar_PLE_LEDE_model_params, m, z, lg_phi):

    assert quasar_hybrid_lumfunc(m, z, **quasar_PLE_LEDE_model_params) \
        == pytest.approx(lg_phi, abs=0.01)


def test_quasar_PLE_model_constraint(quasar_PLE_model_params):

    assert quasar_PLE_model_constraint(**quasar_PLE_model_params)

    quasar_PLE_model_params.update({
        r'\alpha_\mathrm{l}': quasar_PLE_model_params[r'\beta_\mathrm{l}'],
        r'\beta_\mathrm{l}': quasar_PLE_model_params[r'\alpha_\mathrm{l}'],
    })
    assert not quasar_PLE_model_constraint(**quasar_PLE_model_params)


def test_quasar_hybrid_model_constraint(quasar_PLE_LEDE_model_params):

    assert quasar_hybrid_model_constraint(**quasar_PLE_LEDE_model_params)

    quasar_PLE_LEDE_model_params.update({
        r'\alpha': quasar_PLE_LEDE_model_params[r'\beta'],
        r'\beta': quasar_PLE_LEDE_model_params[r'\alpha'],
    })
    assert not quasar_hybrid_model_constraint(**quasar_PLE_LEDE_model_params)


class TestLumFuncModeller:

    @pytest.mark.parametrize(
        "z,lumfunc,lumparams,lumvar,threshold,cosmology,value",
        [
            (
                2.0,
                quasar_PLE_lumfunc,
                'quasar_PLE_model_params',
                'magnitude',
                -22.,
                Planck15,
                5.91e-5,
            ),
            (
                2.0,
                alpha_emitter_schechter_lumfunc,
                'alpha_emitter_Schechter_model_params',
                'luminosity',
                3.e-16,
                Planck15,
                5.63e-5,
            ),
        ]
    )
    def test_comoving_number_density(self, quasar_PLE_model_params,
                                     alpha_emitter_Schechter_model_params,
                                     z, lumfunc, lumparams, lumvar, threshold,
                                     cosmology, value):

        lumparams = locals()[lumparams]

        test_instance = LumFuncModeller(
            lumfunc, lumparams, lumvar, threshold, cosmology
        )

        assert test_instance.comoving_number_density(z) \
            == pytest.approx(value, rel=0.01)


    @pytest.mark.parametrize(
        "z,lumfunc,lumparams,lumvar,threshold,cosmology,value",
        [
            (
                2.0,
                quasar_PLE_lumfunc,
                'quasar_PLE_model_params',
                'magnitude',
                -22.,
                Planck15,
                -0.332,
            ),
            (
                2.0,
                alpha_emitter_schechter_lumfunc,
                'alpha_emitter_Schechter_model_params',
                'luminosity',
                3.e-16,
                Planck15,
                8.19,
            ),
        ]
    )
    def test_evolution_bias(self, quasar_PLE_model_params,
                            alpha_emitter_Schechter_model_params,
                            z, lumfunc, lumparams, lumvar, threshold,
                            cosmology, value):

        lumparams = locals()[lumparams]

        test_instance = LumFuncModeller(
            lumfunc, lumparams, lumvar, threshold, cosmology
        )

        assert test_instance.evolution_bias(z) \
            == pytest.approx(value, rel=0.01)


    @pytest.mark.parametrize(
        "z,lumfunc,lumparams,lumvar,threshold,cosmology,value",
        [
            (
                2.0,
                quasar_PLE_lumfunc,
                'quasar_PLE_model_params',
                'magnitude',
                -22.,
                Planck15,
                0.0748,
            ),
            (
                2.0,
                alpha_emitter_schechter_lumfunc,
                'alpha_emitter_Schechter_model_params',
                'luminosity',
                3.e-16,
                Planck15,
                0.54,
            ),
        ]
    )
    def test_magnification_bias(self, quasar_PLE_model_params,
                                alpha_emitter_Schechter_model_params,
                                z, lumfunc, lumparams, lumvar, threshold,
                                cosmology, value):

        lumparams = locals()[lumparams]

        test_instance = LumFuncModeller(
            lumfunc, lumparams, lumvar, threshold, cosmology
        )

        assert test_instance.magnification_bias(z) \
            == pytest.approx(value, rel=0.01)
