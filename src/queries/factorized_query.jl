
"""
Only active latents are tracked
"""
@gen function prior(q::FactorizedQuery{L,C,O} where {L, C, O})
    new_context = copy(q.context)
    for (latent, dist, active) in zip(q.latents, q.distributions,
                              q.active)
        if active
            new_value = @trace(sample(dist), latent)
        else
            new_value = sample(dist)
        update_context!(new_context, latent, new_value)
        end
    end
    return new_context
end
