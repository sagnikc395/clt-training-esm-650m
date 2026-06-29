Hi Sagnik, I'm trying to steer the project's problem statement due to some recent publications, so I'll let you know of some starting points by tonight.
5 repliesSagnik Chatterjee  [10:29 PM]
hi , any updates for the same 
Saishradha Mohanty  [9:03 AM]
An initial goal that I can think of right now would be to optimize the training of CLTs on all the forward layers using optimization methods of the architecture existing in the literature. I had trained an initial CLT on ESM-2 650M, however, it takes very long to train. We want to ensure that our optimized CLT methods can be translated to the bigger ESM-C 3-6b models also that were released by Biohub recently. These were the inital CLT that I had trained inspired from Eleuther AI - /project/pi_annagreen_umass_edu/saishradha/plm_circuits_private/clt-training-optimized (edited) 
Saishradha Mohanty  [10:01 AM]
Maybe ideas on the sparse computation side, LoRA etc could be helpful sicne for a specific downstream tasks it is very clear that these foundation models use a very sparse circuit.
Sagnik Chatterjee  [10:04 AM]
hmm , makes sense , ill think from lora angle
Saishradha Mohanty  [1:05 PM]
My suggestion would be to first try and identify what is making the exisiting ElutherAI or my CLT training setup slow, like what is the main bottleneck and then what aspects of the pLM / architecture can be used to make the training more efficient or optimized.


