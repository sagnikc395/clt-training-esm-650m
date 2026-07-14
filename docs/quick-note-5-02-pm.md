# clt traininig (SAGE LAB work)

**Date:** Jun 30, 2026 at 5:02 PM — 5:09 PM

## Notes

*No notes.*

## Transcript

**[00:00] Me:** (speaking in foreign language)  
**[00:00] Others:** Yeah, this paper which does apply CLTs for protein language models, but it's very, very  
**[00:30] Me:** (people chattering)  
**[00:30] Others:** as in it is not it's for a supervised task however we want to see whether it works for an unsupervised task or not so yeah my so the first step could be like the unsupervised task is something that I am currently writing the paper about so So in the meantime, what you can do is try and apply  
**[01:00] Me:** Yeah, it's in the same like the parent directory heart like the PLT circuits and like one was like the you have sent the optimized code. So from what I could understand it was like it was analyzing the sweep. So you have like use one DB to like store the intermediate results and we're like analyzing this from there and  
**[01:00] Others:** the CLTs and I had also trained PLTs there in the same surrounding directories. I can send the code for that.  
**[01:30] Me:** trying to optimize from that right.  
**[01:30] Others:** Yeah, so one direction could be that you can see, you can go and check whether the CLTs actually, first of all, are they able to identify the same circuits, like the ones that the paper for the protein that this particular paper does. So you can look at the.  
**[02:00] Me:** (people chattering)  
**[02:00] Others:** that they use for the supervised prediction task, mutation effect prediction task. And I think they do it for two proteins. And yeah, you can go and check whether our current CLTs are able to at least generate the same mode circuits or not. I'm.  
**[02:30] Me:** (indistinct chatter)  
**[02:30] Others:** By hypothesis, they should be able to do that. And or if they are able to generate even more features than what their CLTs do, because they trained CLTs on a 35 million model, but our CLT is trained on 650 million model. So it should be more powerful. That is like the first sanity check. So once you do that, I think what we can do is go and try to apply it for a  
**[03:00] Me:** Okay, yeah, that makes sense. Also on the optimization side, do you think like from what I like, I don't have much idea about optimizing those runs on GPU.  
**[03:00] Others:** an unsupervised mutation effect prediction task. So for that, there needs to be a slight change. So right now you can just go and apply ours to this. That's probably the first step. Yeah. Thank you.  
**[03:30] Me:** Do you think like should I go and read more about optimization stuff or It's like do we like we need to go like how to do like proper like with Like how to like like do the optimizations better See you.  
**[03:30] Others:** By optimization, it would be very useful for any kind of model, even for you, for a later career to know how to optimize the models. like the current methods of using flash attention, Triton kernels or something.  
**[04:00] Me:** So, I was thinking more on the Mykon top side, like I'm not biased towards anything but like I'm just thinking from like a practical standpoint, let's say like how much time It will take me to learn that because like let's  
**[04:00] Others:** that would be very useful to have because the bigger models are extremely expensive to use and it's always good to know how to use the resources very conveniently. So it depends on what direction you want to take. Do you want to look more...  
**[04:30] Me:** say there's a bigger problem we are trying to solve. I know you have to learn many things, but again, the time sink to learn that to solve the bigger problem would be kind of huge. I was thinking something like Triton or something where you can write the kernels in Python, and I think the compiler's converted into optimized code or code, something like that. But I was not sure about that part. Wanted to get your perspective also on that.  
**[04:30] Others:** [ Silence ]  
**[05:00] Me:** (speaking foreign language)  
**[05:00] Others:** Yeah, like if you want to look at more at the Meccan top side, then you can go ahead and straight away try and use the CLTs, apply CLTs on these proteins and see what kind of circuits you get and what kind of features you get out of those. And if they look the same or they look different than the ones that they find to start with for the same supervised mutation of a prediction task. And later we can translate to the.  
**[05:30] Me:** Okay, okay, right. I'll just start on the other. I think I'll just start from the applying the CLT part first and like going through the optimization. I went through the, like, to be very honest, like...  
**[05:30] Others:** unsupervised mutation effect prediction tasks to see how well they do in the bigger model like 650 million. Yeah. Yeah.  
**[06:00] Me:** I was not like I don't know much about like optimizing the kernel. So I was just reading the docs and it was not leading anywhere. So it was kind of like, you know, panic mode. So like, so that's yeah. Yeah, like I'll probably I can work on it, but like the later time, like I want to get some work done then like, you know, think a little. Yeah. So I wanted to get something out of my, like my work here. Yeah. So that's why.  
**[06:00] Others:** Yeah, that's okay you can let it like leave it leave that part, yeah.  
**[06:30] Me:** I'll try to apply the CLT thing like you said so that would be a little helpful I think like I'll show some work Yeah, so I think yeah, that's the only thing I was a little confused about not particularly so okay I'll give you a bit like is it fine if I give you an update on Tuesdays during this time Okay, okay fine. I'll give you an update  
**[06:30] Others:** Yeah, not a problem. Okay. Yeah, sure. Okay. Okay. Yeah, sure.  
**[07:00] Me:** on a weekly basis on next Tuesday I guess. Okay, perfect. Yeah, thank you so much for this. Thank you. Yeah, bye.  
**[07:00] Others:** Yeah. Okay, bye. (sonic logo)  
